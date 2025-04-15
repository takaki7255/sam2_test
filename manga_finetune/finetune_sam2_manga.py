#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2モデルを漫画画像セグメンテーション用にファインチューニングするためのスクリプト
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from PIL import Image
import random
import argparse
import logging
import datetime

# SAM2のインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sam2.build_sam import sam2_model_registry
from sam2.utils.transforms import ResizeLongestSide

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MangaSegmentationDataset(Dataset):
    """漫画セグメンテーションデータセット"""

    def __init__(self, root_dir, transform=None, image_size=1024, is_train=True, split=0.8):
        """
        Args:
            root_dir: データセットのルートディレクトリ
            transform: 適用する変換
            image_size: モデル入力用のサイズ
            is_train: 訓練用かテスト用か
            split: 訓練/検証の分割比率
        """
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.anno_dir = self.root_dir / "annotations"
        self.transform = transform
        self.image_size = image_size
        self.is_train = is_train
        
        # 画像ファイルのリストを取得
        self.image_files = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
        
        # 訓練/検証で分割
        random.seed(42)  # 再現性のため
        random.shuffle(self.image_files)
        split_idx = int(len(self.image_files) * split)
        
        if is_train:
            self.image_files = self.image_files[:split_idx]
        else:
            self.image_files = self.image_files[split_idx:]
        
        logger.info(f"{'Training' if is_train else 'Validation'} dataset created with {len(self.image_files)} images")
        
        # SAM2の前処理
        self.transform = ResizeLongestSide(image_size)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.anno_dir / f"{img_path.stem}.png"  # アノテーションは.pngとする
        
        # 画像の読み込み
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # マスクの読み込み
        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert("L"))
            # 2値化（必要に応じて）
            mask = (mask > 128).astype(np.float32)
        else:
            logger.warning(f"Mask not found for {img_path.name}, using empty mask")
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # ポイントプロンプトの生成（マスクからランダムにポイントを選択）
        y_indices, x_indices = np.where(mask > 0.5)
        if len(y_indices) > 0:
            # マスク内から1つポイントを選択
            random_idx = np.random.randint(0, len(y_indices))
            point_coords = np.array([[x_indices[random_idx], y_indices[random_idx]]])
            point_labels = np.array([1])  # 1はフォアグラウンド
        else:
            # マスクが空の場合はランダムなポイントを選択
            h, w = mask.shape
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            point_coords = np.array([[x, y]])
            point_labels = np.array([0])  # 0はバックグラウンド
        
        # SAM2の前処理
        image_resized = self.transform.apply_image(image)
        image_tensor = torch.as_tensor(image_resized.transpose(2, 0, 1)).float()
        
        # マスクのリサイズ
        mask_resized = cv2.resize(mask, (image_resized.shape[1], image_resized.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.as_tensor(mask_resized).float().unsqueeze(0)
        
        # ポイントの座標を変換
        point_coords_resized = self.transform.apply_coords(point_coords, (image.shape[0], image.shape[1]))
        point_coords_tensor = torch.as_tensor(point_coords_resized).float()
        point_labels_tensor = torch.as_tensor(point_labels).long()
        
        # 元の画像サイズ
        original_size = torch.tensor([image.shape[0], image.shape[1]])
        
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "point_coords": point_coords_tensor,
            "point_labels": point_labels_tensor,
            "original_size": original_size,
            "image_path": str(img_path)
        }


def calculate_loss(pred_masks, gt_masks, focal_weight=20.0, dice_weight=1.0):
    """
    損失関数の計算：Focal LossとDice Lossの組み合わせ
    """
    # Focal Loss
    def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2):
        prob = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        
        return loss.mean()
    
    # Dice Loss
    def dice_loss(inputs, targets, smooth=1.0):
        prob = torch.sigmoid(inputs)
        
        # フラット化
        inputs_flat = prob.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_coef = (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        
        return 1 - dice_coef
    
    focal = sigmoid_focal_loss(pred_masks, gt_masks)
    dice = dice_loss(pred_masks, gt_masks)
    
    loss = focal_weight * focal + dice_weight * dice
    return loss, focal, dice


def train_one_epoch(model, dataloader, optimizer, device, epoch, config):
    """1エポック分の訓練を実行"""
    model.train()
    total_loss = 0.0
    total_focal_loss = 0.0
    total_dice_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        # データの準備
        image = batch["image"].to(device)
        mask_gt = batch["mask"].to(device)
        point_coords = batch["point_coords"].to(device)
        point_labels = batch["point_labels"].to(device)
        original_size = batch["original_size"].to(device)
        
        # 点プロンプトでの予測
        outputs = model(
            image,
            point_coords, 
            point_labels,
            multimask_output=False  # 単一マスク出力
        )
        
        # 損失の計算
        loss, focal_loss, dice_loss = calculate_loss(
            outputs["low_res_logits"], 
            F.interpolate(mask_gt, size=outputs["low_res_logits"].shape[-2:], mode='bilinear', align_corners=False),
            focal_weight=config["loss"]["focal_weight"],
            dice_weight=config["loss"]["dice_weight"]
        )
        
        # 勾配の計算と更新
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # 損失の記録
        total_loss += loss.item()
        total_focal_loss += focal_loss.item()
        total_dice_loss += dice_loss.item()
        
        pbar.set_postfix({
            "loss": loss.item(), 
            "focal": focal_loss.item(), 
            "dice": dice_loss.item()
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_focal_loss = total_focal_loss / len(dataloader)
    avg_dice_loss = total_dice_loss / len(dataloader)
    
    return avg_loss, avg_focal_loss, avg_dice_loss


def validate(model, dataloader, device, config):
    """検証データでの評価"""
    model.eval()
    total_loss = 0.0
    total_focal_loss = 0.0
    total_dice_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # データの準備
            image = batch["image"].to(device)
            mask_gt = batch["mask"].to(device)
            point_coords = batch["point_coords"].to(device)
            point_labels = batch["point_labels"].to(device)
            original_size = batch["original_size"].to(device)
            
            # 点プロンプトでの予測
            outputs = model(
                image,
                point_coords, 
                point_labels,
                multimask_output=False
            )
            
            # 損失の計算
            loss, focal_loss, dice_loss = calculate_loss(
                outputs["low_res_logits"], 
                F.interpolate(mask_gt, size=outputs["low_res_logits"].shape[-2:], mode='bilinear', align_corners=False),
                focal_weight=config["loss"]["focal_weight"],
                dice_weight=config["loss"]["dice_weight"]
            )
            
            # 損失の記録
            total_loss += loss.item()
            total_focal_loss += focal_loss.item()
            total_dice_loss += dice_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_focal_loss = total_focal_loss / len(dataloader)
    avg_dice_loss = total_dice_loss / len(dataloader)
    
    return avg_loss, avg_focal_loss, avg_dice_loss


def visualize_prediction(model, dataset, device, output_dir, num_samples=5):
    """予測結果の可視化"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            
            # データの準備
            image = sample["image"].unsqueeze(0).to(device)
            mask_gt = sample["mask"].to(device)
            point_coords = sample["point_coords"].unsqueeze(0).to(device)
            point_labels = sample["point_labels"].unsqueeze(0).to(device)
            original_size = sample["original_size"].to(device)
            
            # 予測
            outputs = model(
                image,
                point_coords, 
                point_labels,
                multimask_output=False
            )
            
            # マスクの取得
            mask_pred = torch.sigmoid(outputs["masks"]).cpu().numpy()[0, 0]
            
            # 可視化
            image_np = image.cpu().numpy()[0].transpose(1, 2, 0)
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
            
            mask_gt_np = mask_gt.cpu().numpy()[0, 0]
            
            # 点プロンプトの座標
            point_coords_np = point_coords.cpu().numpy()[0]
            point_labels_np = point_labels.cpu().numpy()[0]
            
            # プロット
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 元画像
            axes[0].imshow(image_np)
            for p_idx, (x, y) in enumerate(point_coords_np):
                color = 'green' if point_labels_np[p_idx] == 1 else 'red'
                axes[0].scatter(x, y, color=color, s=80, marker='*')
            axes[0].set_title("Input Image with Points")
            axes[0].axis("off")
            
            # 正解マスク
            axes[1].imshow(image_np)
            axes[1].imshow(mask_gt_np, alpha=0.5, cmap='Reds')
            axes[1].set_title("Ground Truth Mask")
            axes[1].axis("off")
            
            # 予測マスク
            axes[2].imshow(image_np)
            axes[2].imshow(mask_pred > 0.5, alpha=0.5, cmap='Blues')
            axes[2].set_title("Predicted Mask")
            axes[2].axis("off")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"vis_sample_{i}.png"))
            plt.close()


def main(config_path):
    """メイン関数"""
    # 設定ファイルの読み込み
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # モデルの構築
    model_type = config["model"]["type"]
    backbone = config["model"]["backbone"]
    checkpoint = config["model"]["checkpoint"]
    
    logger.info(f"Building {model_type} model with {backbone} backbone...")
    model = sam2_model_registry[backbone](checkpoint=checkpoint)
    model.to(device)
    
    # データセットとデータローダーの設定
    image_size = config["data"]["image_size"]
    dataset_dir = config["data"]["dataset_dir"]
    
    train_dataset = MangaSegmentationDataset(
        root_dir=dataset_dir,
        image_size=image_size,
        is_train=True
    )
    
    valid_dataset = MangaSegmentationDataset(
        root_dir=dataset_dir,
        image_size=image_size,
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # オプティマイザとスケジューラの設定
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    num_epochs = config["training"]["num_epochs"]
    warmup_steps = config["training"]["warmup_steps"]
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # ウォームアップとコサインスケジューラの組み合わせ
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    
    # 出力ディレクトリの設定
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"manga_sam2_finetune_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 訓練ループ
    logger.info("Starting training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 訓練
        train_loss, train_focal, train_dice = train_one_epoch(
            model, train_loader, optimizer, device, epoch, config
        )
        
        # 検証
        val_loss, val_focal, val_dice = validate(model, valid_loader, device, config)
        
        # スケジューラの更新
        scheduler.step()
        
        # ログの出力
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Train Loss: {train_loss:.4f} (Focal: {train_focal:.4f}, Dice: {train_dice:.4f}) - "
                   f"Val Loss: {val_loss:.4f} (Focal: {val_focal:.4f}, Dice: {val_dice:.4f})")
        
        # 損失の記録
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # モデルの保存
        if val_loss < best_val_loss or (epoch + 1) % config["training"]["save_interval"] == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(output_dir, "best_model.pth")
            else:
                save_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pth")
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, save_path)
            
            logger.info(f"Model saved to {save_path}")
    
    # 学習曲線の可視化
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    
    # 予測結果の可視化
    vis_dir = os.path.join(output_dir, "visualizations")
    visualize_prediction(model, valid_dataset, device, vis_dir)
    
    logger.info(f"Training completed. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 Manga Segmentation Finetuning")
    parser.add_argument("--config", type=str, default="sam2_manga_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    main(args.config)

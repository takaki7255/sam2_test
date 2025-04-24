import os
import json
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry, SamPredictor
from peft import get_peft_model, LoraConfig, TaskType
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

# カスタムデータセットクラス
class MangaSegmentationDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None):
        """
        Args:
            json_path: アノテーションJSONファイルのパス
            img_dir: 画像ディレクトリのパス
            transform: 画像の前処理
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # 画像とアノテーションのペアのみ抽出
        self.valid_entries = []
        for img in self.data['images']:
            if img['annotated']:
                self.valid_entries.append(img)
        
        self.img_dir = img_dir
        self.transform = transform
        
        # アノテーションデータを取得
        self.annotations = self.data.get('annotations', [])
        
        # カテゴリー情報の取得（コマ=1, 吹き出し=2など）
        self.categories = {cat['id']: cat['name'] for cat in self.data.get('categories', [])}
    
    def __len__(self):
        return len(self.valid_entries)
    
    def __getitem__(self, idx):
        img_info = self.valid_entries[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # 画像の読み込み
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # この画像に関連するアノテーションを取得
        img_annotations = [anno for anno in self.annotations if anno['image_id'] == img_id]
        
        # マスクの作成 (カテゴリごとに別々のマスク)
        masks = {}
        for cat_id in self.categories.keys():
            masks[cat_id] = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        
        # アノテーションからマスクを生成
        for anno in img_annotations:
            cat_id = anno['category_id']
            segmentation = anno['segmentation']
            
            # 多角形のセグメンテーションデータをマスクに変換
            for seg in segmentation:
                pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(masks[cat_id], [pts], 1)
        
        # プロンプトポイントの生成（マスク内の中心点を使用）
        prompt_points = []
        prompt_labels = []
        
        for cat_id, mask in masks.items():
            if np.any(mask):
                # マスクの中心点を計算
                moments = cv2.moments(mask)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    prompt_points.append([cx, cy])
                    prompt_labels.append(1)  # 前景ポイント
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'masks': masks,
            'prompt_points': np.array(prompt_points) if prompt_points else np.zeros((0, 2)),
            'prompt_labels': np.array(prompt_labels) if prompt_labels else np.zeros(0),
            'image_id': img_id,
            'original_size': (img_info['height'], img_info['width'])
        }


# SAM2モデルにLoRAを適用するための設定
def create_lora_sam2(model):
    """SAM2モデルにLoRAを適用"""
    config = LoraConfig(
        r=16,  # LoRAのランク
        lora_alpha=32,
        target_modules=["query", "value"],  # 注意機構のクエリと値の投影に適用
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    
    lora_model = get_peft_model(model, config)
    return lora_model


# 損失関数の定義
class SegmentationLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha  # Dice損失とIoU損失の重み付け
    
    def forward(self, pred_masks, gt_masks):
        # 平滑化係数
        smooth = 1e-5
        
        # バイナリクロスエントロピー損失
        bce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks)
        
        # シグモイド関数で確率に変換
        pred_prob = torch.sigmoid(pred_masks)
        
        # Dice損失
        intersection = torch.sum(pred_prob * gt_masks, dim=(1, 2, 3))
        union = torch.sum(pred_prob, dim=(1, 2, 3)) + torch.sum(gt_masks, dim=(1, 2, 3))
        dice_loss = 1 - (2. * intersection + smooth) / (union + smooth)
        
        # IoU損失
        intersection = torch.sum(pred_prob * gt_masks, dim=(1, 2, 3))
        union = torch.sum(pred_prob, dim=(1, 2, 3)) + torch.sum(gt_masks, dim=(1, 2, 3)) - intersection
        iou_loss = 1 - (intersection + smooth) / (union + smooth)
        
        # 組み合わせた損失
        combined_loss = self.alpha * dice_loss.mean() + (1 - self.alpha) * iou_loss.mean() + bce_loss
        
        return combined_loss


# トレーニングループ
def train_sam2_for_manga(model, train_loader, val_loader, device, num_epochs=10):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = SegmentationLoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # トレーニングフェーズ
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in progress_bar:
            images = batch['image'].to(device)
            masks = {k: v.to(device) for k, v in batch['masks'].items()}
            points = batch['prompt_points'].to(device)
            labels = batch['prompt_labels'].to(device)
            
            # 入力画像の前処理
            image_embeddings = model.image_encoder(images)
            
            loss_batch = 0
            # 各カテゴリーについて処理
            for cat_id, gt_mask in masks.items():
                if not torch.any(gt_mask):
                    continue
                
                # プロンプトエンコーディング
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=points,
                    boxes=None,
                    masks=None,
                )
                
                # マスク予測
                mask_predictions, _ = model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                
                # 損失計算
                loss = criterion(mask_predictions, gt_mask.unsqueeze(1).float())
                loss_batch += loss
            
            # 逆伝播と最適化
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            
            train_loss += loss_batch.item()
            progress_bar.set_postfix({'loss': loss_batch.item()})
        
        train_loss /= len(train_loader)
        
        # バリデーションフェーズ
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(device)
                masks = {k: v.to(device) for k, v in batch['masks'].items()}
                points = batch['prompt_points'].to(device)
                labels = batch['prompt_labels'].to(device)
                
                image_embeddings = model.image_encoder(images)
                
                loss_batch = 0
                for cat_id, gt_mask in masks.items():
                    if not torch.any(gt_mask):
                        continue
                    
                    sparse_embeddings, dense_embeddings = model.prompt_encoder(
                        points=points,
                        boxes=None,
                        masks=None,
                    )
                    
                    mask_predictions, _ = model.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    
                    loss = criterion(mask_predictions, gt_mask.unsqueeze(1).float())
                    loss_batch += loss
                
                val_loss += loss_batch.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 最良モデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, 'best_sam2_manga_model.pth')
    
    return model


# メイン実行関数
def main():
    # パラメータ設定
    json_path = 'manga_annotations.json'  # JSONアノテーションファイル
    img_dir = '/datasets/manga-frame/'    # 画像ディレクトリ
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    num_epochs = 20
    
    # データセットの作成
    full_dataset = MangaSegmentationDataset(json_path, img_dir)
    
    # 訓練データとバリデーションデータに分割
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # データローダーの作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # SAM2モデルのロード
    sam2_model = sam_model_registry["vit_h"](checkpoint="sam2_hq.pth")
    
    # 画像エンコーダを凍結
    for param in sam2_model.image_encoder.parameters():
        param.requires_grad = False
    
    # LoRAを適用
    lora_sam2_model = create_lora_sam2(sam2_model)
    
    # モデルのトレーニング
    trained_model = train_sam2_for_manga(lora_sam2_model, train_loader, val_loader, device, num_epochs)
    
    # 最終モデルの保存
    torch.save({
        'model_state_dict': trained_model.state_dict(),
    }, 'final_sam2_manga_model.pth')
    
    print("トレーニング完了!")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2のファインチューニング済みモデルを評価するスクリプト
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from PIL import Image
import argparse
import logging
from skimage.metrics import adapted_rand_error, variation_of_information

# SAM2のインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sam2.build_sam import sam2_model_registry
from sam2.utils.transforms import ResizeLongestSide

# 自作モジュールのインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from finetune_sam2_manga import MangaSegmentationDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_metrics(pred_masks, gt_masks):
    """
    セグメンテーション評価指標の計算
    """
    metrics = {}
    
    # 予測マスクのバイナリ化
    pred_binary = (pred_masks > 0.5).astype(np.uint8)
    gt_binary = (gt_masks > 0.5).astype(np.uint8)
    
    # IoU (Intersection over Union)
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    iou = intersection / union if union > 0 else 0.0
    metrics["IoU"] = iou
    
    # Dice係数
    dice = (2 * intersection) / (pred_binary.sum() + gt_binary.sum()) if (pred_binary.sum() + gt_binary.sum()) > 0 else 0.0
    metrics["Dice"] = dice
    
    # 精度 (Precision)
    precision = intersection / pred_binary.sum() if pred_binary.sum() > 0 else 0.0
    metrics["Precision"] = precision
    
    # 再現率 (Recall)
    recall = intersection / gt_binary.sum() if gt_binary.sum() > 0 else 0.0
    metrics["Recall"] = recall
    
    # F1スコア
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    metrics["F1"] = f1
    
    # 誤差指標：適応RAND誤差
    try:
        are_score, _ = adapted_rand_error(gt_binary, pred_binary)
        metrics["ARE"] = are_score
    except:
        metrics["ARE"] = float('nan')
    
    # 変分情報
    try:
        vi_score, _, _ = variation_of_information(gt_binary, pred_binary)
        metrics["VI"] = vi_score
    except:
        metrics["VI"] = float('nan')
    
    # 境界F値の計算
    # 漫画特有の境界線評価
    try:
        # エッジ検出
        gt_edges = cv2.Canny(gt_binary * 255, 100, 200)
        pred_edges = cv2.Canny(pred_binary * 255, 100, 200)
        
        # 境界線の一致を評価
        matched_edges = np.logical_and(gt_edges > 0, pred_edges > 0).sum()
        boundary_precision = matched_edges / pred_edges.sum() if pred_edges.sum() > 0 else 0.0
        boundary_recall = matched_edges / gt_edges.sum() if gt_edges.sum() > 0 else 0.0
        boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall) if (boundary_precision + boundary_recall) > 0 else 0.0
        
        metrics["Boundary_F1"] = boundary_f1
    except:
        metrics["Boundary_F1"] = float('nan')
    
    return metrics


def evaluate_model(model, dataloader, device, output_dir):
    """モデルの評価を実行"""
    model.eval()
    
    all_metrics = {
        "IoU": [],
        "Dice": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "ARE": [],
        "VI": [],
        "Boundary_F1": []
    }
    
    # 可視化用のディレクトリ
    vis_dir = os.path.join(output_dir, "evaluation_vis")
    os.makedirs(vis_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # データの準備
            image = batch["image"].to(device)
            mask_gt = batch["mask"]
            point_coords = batch["point_coords"].to(device)
            point_labels = batch["point_labels"].to(device)
            image_path = batch["image_path"]
            
            # 予測
            outputs = model(
                image,
                point_coords, 
                point_labels,
                multimask_output=False
            )
            
            # マスクの取得
            mask_pred = torch.sigmoid(outputs["masks"]).cpu().numpy()
            
            # バッチ内の各サンプルに対して評価
            for b_idx in range(image.shape[0]):
                curr_pred = mask_pred[b_idx, 0]
                curr_gt = mask_gt[b_idx, 0].numpy()
                
                # 評価指標の計算
                metrics = calculate_metrics(curr_pred, curr_gt)
                
                # 指標の保存
                for metric_name, value in metrics.items():
                    all_metrics[metric_name].append(value)
                
                # 定期的に可視化（例：20サンプルごと）
                if i * image.shape[0] + b_idx < 100 and (i * image.shape[0] + b_idx) % 5 == 0:
                    # 画像の復元
                    curr_image = image[b_idx].cpu().numpy().transpose(1, 2, 0)
                    curr_image = (curr_image - curr_image.min()) / (curr_image.max() - curr_image.min())
                    
                    # 点プロンプトの座標
                    curr_points = point_coords[b_idx].cpu().numpy()
                    curr_labels = point_labels[b_idx].cpu().numpy()
                    
                    # 可視化
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    
                    # 入力画像
                    axes[0].imshow(curr_image)
                    for p_idx, (x, y) in enumerate(curr_points):
                        color = 'green' if curr_labels[p_idx] == 1 else 'red'
                        axes[0].scatter(x, y, color=color, s=80, marker='*')
                    axes[0].set_title("Input Image with Points")
                    axes[0].axis("off")
                    
                    # 正解マスク
                    axes[1].imshow(curr_image)
                    axes[1].imshow(curr_gt, alpha=0.5, cmap='Reds')
                    axes[1].set_title("Ground Truth Mask")
                    axes[1].axis("off")
                    
                    # 予測マスク
                    axes[2].imshow(curr_image)
                    axes[2].imshow(curr_pred > 0.5, alpha=0.5, cmap='Blues')
                    metrics_text = f"IoU: {metrics['IoU']:.4f}, Dice: {metrics['Dice']:.4f}"
                    axes[2].set_title(f"Predicted Mask\n{metrics_text}")
                    axes[2].axis("off")
                    
                    plt.tight_layout()
                    img_name = os.path.basename(image_path[b_idx])
                    plt.savefig(os.path.join(vis_dir, f"eval_{i}_{b_idx}_{img_name}.png"))
                    plt.close()
    
    # 平均値の計算
    avg_metrics = {metric: np.mean(values) for metric, values in all_metrics.items()}
    
    # 結果の表示
    logger.info("Evaluation Results:")
    for metric, value in avg_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # グラフの生成
    plot_metrics_histograms(all_metrics, output_dir)
    
    return avg_metrics


def plot_metrics_histograms(metrics, output_dir):
    """評価指標のヒストグラムをプロット"""
    os.makedirs(output_dir, exist_ok=True)
    
    for metric_name, values in metrics.items():
        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=20, alpha=0.7, color='blue')
        plt.xlabel(metric_name)
        plt.ylabel('Frequency')
        plt.title(f'{metric_name} Distribution')
        plt.grid(True, alpha=0.3)
        plt.axvline(np.mean(values), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(values):.4f}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{metric_name}_histogram.png"))
        plt.close()


def main(args):
    """メイン関数"""
    # 設定ファイルの読み込み
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # モデルの構築
    backbone = config["model"]["backbone"]
    checkpoint = args.checkpoint
    
    logger.info(f"Loading model with {backbone} backbone from {checkpoint}...")
    model = sam2_model_registry[backbone]()
    checkpoint_data = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.to(device)
    
    # データセットとデータローダーの設定
    image_size = config["data"]["image_size"]
    dataset_dir = config["data"]["dataset_dir"]
    
    test_dataset = MangaSegmentationDataset(
        root_dir=dataset_dir,
        image_size=image_size,
        is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 出力ディレクトリの設定
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # モデルの評価
    logger.info("Starting evaluation...")
    avg_metrics = evaluate_model(model, test_loader, device, output_dir)
    
    # 結果の保存
    result_path = os.path.join(output_dir, "evaluation_results.txt")
    with open(result_path, "w") as f:
        for metric, value in avg_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    logger.info(f"Evaluation completed. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 Manga Segmentation Evaluation")
    parser.add_argument("--config", type=str, default="../manga_finetune/sam2_manga_config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="./manga_sam2_eval", help="Path to output directory")
    args = parser.parse_args()
    
    main(args)

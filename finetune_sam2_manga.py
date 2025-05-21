"""
SAM2モデルをマンガセグメンテーション用にファインチューニングするスクリプト

このスクリプトは、Segment Anything Model 2 (SAM2)をマンガ画像のセグメンテーション向けに
ファインチューニングするためのものです。LoRAアダプターを使用して、モデルの一部のみを
効率的に学習させることで、マンガ画像に特化したセグメンテーション性能を向上させます。

参考: https://datascientistsdiary.com/fine-tuning-sam-2/
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import traceback
import inspect
import albumentations as A
from torch.nn import functional as F
from peft import get_peft_model, LoraConfig, TaskType
import datetime
import logging
import sys

# SAM2のビルド関数をインポート
from sam2.build_sam import build_sam2

# ロギング設定
def setup_logger(log_file=None):
    """ロギングの設定を行う関数"""
    logger = logging.getLogger('sam2_finetune')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # コンソール出力用のハンドラー
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ファイル出力用のハンドラー（指定された場合）
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class ResizeLongestSide:
    """
    画像の長辺を指定サイズにリサイズし、アスペクト比を維持する処理
    必要に応じてパディングを行い、固定サイズの出力にする
    """
    def __init__(self, target_length: int):
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        画像をリサイズしてパディングを追加
        """
        h, w = image.shape[:2]
        scale = self.target_length / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # リサイズ
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # パディング（ターゲットサイズまで）
        padded = np.zeros((self.target_length, self.target_length, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        return padded

    def apply_coords(self, coords: np.ndarray, original_size: tuple) -> np.ndarray:
        """
        座標をリサイズ後のスケールに変換
        """
        h, w = original_size
        scale = self.target_length / max(h, w)
        coords = coords.copy()
        coords[:, 0] = coords[:, 0] * scale
        coords[:, 1] = coords[:, 1] * scale
        return coords
        
    def apply_mask(self, mask: np.ndarray, original_size: tuple) -> np.ndarray:
        """
        マスクをリサイズしてパディングを追加
        """
        h, w = original_size
        scale = self.target_length / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # リサイズ（最近傍補間を使用してバイナリマスクの性質を保持）
        resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # パディング
        padded = np.zeros((self.target_length, self.target_length), dtype=mask.dtype)
        padded[:new_h, :new_w] = resized
        
        return padded


class MangaSegmentationDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None, img_size=1024):
        """
        マンガセグメンテーション用のデータセット
        
        Args:
            json_path: COCOスタイルのアノテーションJSONファイルパス
            img_dir: 画像ディレクトリパス
            transform: 追加の変換処理（オプション）
            img_size: 入力画像サイズ
        """
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size
        self.resizer = ResizeLongestSide(img_size)
        
        # JSONデータを読み込む
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 画像とアノテーションの対応付け
        self.images = data['images']
        self.annotations = data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in data['categories']}
        
        # 画像IDからアノテーションへのマッピングを作成
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 画像情報の取得
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # 画像の読み込み
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]  # (height, width)
        
        # 画像のリサイズ
        resized_image = self.resizer.apply_image(image)
        
        # この画像に対応するアノテーションを取得
        img_annotations = self.img_to_anns.get(img_id, [])
        
        # カテゴリごとにマスクを準備
        masks = {}
        for cat_id in self.categories.keys():
            masks[cat_id] = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        # 元のサイズでマスクを一時的に作成
        original_masks = {}
        for cat_id in self.categories.keys():
            original_masks[cat_id] = np.zeros(original_size, dtype=np.uint8)
        
        # アノテーションからマスクを生成（元のサイズ）
        for anno in img_annotations:
            cat_id = anno['category_id']
            segmentation = anno.get('segmentation', [])
            
            # セグメンテーションがポリゴン形式の場合
            if isinstance(segmentation, list) and len(segmentation) > 0:
                if isinstance(segmentation[0], list):  # ポリゴンのリスト
                    for polygon in segmentation:
                        # ポリゴンを描画してマスクを生成
                        pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(original_masks[cat_id], [pts], 1)
            # RLE形式の場合
            elif isinstance(segmentation, dict) and 'counts' in segmentation:
                from pycocotools import mask as mask_utils
                rle = segmentation
                mask = mask_utils.decode(rle)
                original_masks[cat_id] = np.logical_or(original_masks[cat_id], mask).astype(np.uint8)
        
        # マスクをリサイズしパディングを追加
        for cat_id in self.categories.keys():
            masks[cat_id] = self.resizer.apply_mask(original_masks[cat_id], original_size)
        
        # プロンプトポイントの生成（マスク内の中心点を使用）
        prompt_points = []
        prompt_labels = []
        
        for cat_id, mask in masks.items():
            if np.any(mask):  # マスクが空でない場合
                # マスクの輪郭を検出
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # 輪郭の中心を計算
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # 中心点をプロンプトとして追加（正例）
                        prompt_points.append([cx, cy])
                        prompt_labels.append(1)  # 正例
                        
                        # 任意: 輪郭外の点を負例としても追加可能
                        # マスクから十分離れた点を選択
                        mask_expanded = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=5)
                        mask_border = mask_expanded - mask
                        
                        if np.any(mask_border):
                            y, x = np.where(mask_border > 0)
                            if len(y) > 0:
                                idx = np.random.randint(0, len(y))
                                prompt_points.append([x[idx], y[idx]])
                                prompt_labels.append(0)  # 負例
        
        # 画像をPyTorchテンソルに変換 (FloatTensorに変換して正規化)
        image = torch.from_numpy(resized_image).float().permute(2, 0, 1) / 255.0
        
        # マスクもPyTorchテンソルに変換
        masks_tensor = {}
        for cat_id, mask in masks.items():
            masks_tensor[cat_id] = torch.from_numpy(mask).float()
        
        # トランスフォームがある場合はそれを適用
        if self.transform:
            transformed = self.transform(image=image.permute(1, 2, 0).numpy())
            image = torch.from_numpy(transformed['image']).permute(2, 0, 1)
        
        # プロンプトポイントをテンソルに変換
        prompt_points_tensor = torch.tensor(prompt_points, dtype=torch.float) if prompt_points else torch.zeros((0, 2))
        prompt_labels_tensor = torch.tensor(prompt_labels, dtype=torch.int) if prompt_labels else torch.zeros(0, dtype=torch.int)
        
        return {
            'image': image,
            'masks': masks_tensor,
            'prompt_points': prompt_points_tensor,
            'prompt_labels': prompt_labels_tensor,
            'image_id': img_id,
            'original_size': original_size,
            'input_size': (self.img_size, self.img_size)
        }


# マスク変換ユーティリティ関数
def mask_to_polygons(mask):
    """
    バイナリマスクをCOCOスタイルのポリゴンに変換
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if len(cnt) >= 6:  # 小さすぎる輪郭はフィルタリング
            # 輪郭を簡素化してメモリ使用量を削減
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            poly = approx.squeeze().flatten().tolist()
            if len(poly) >= 6:
                polygons.append(poly)
    return polygons


# 損失関数の定義
class SegmentationLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha  # Dice損失とIoU損失の重み付け
    
    def forward(self, pred_masks, gt_masks):
        """
        予測マスクと正解マスク間の組み合わせ損失を計算
        
        Args:
            pred_masks: 予測マスク (B, 1, H, W)
            gt_masks: 正解マスク (B, 1, H, W)
        """
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


# モデル構造を分析して表示する関数
def print_model_structure(model, max_depth=3):
    """モデルの構造を階層的に表示し、学習可能なパラメータを特定する"""
    print("\n==================== モデル構造の詳細分析 ====================")
    
    # 全パラメータ数と学習可能なパラメータ数をカウントするための変数
    total_params = 0
    total_trainable_params = 0
    param_details = {}
    
    def _print_module(module, prefix="", depth=0, path=""):
        nonlocal total_params, total_trainable_params
        
        if depth > max_depth:
            return
        
        # モジュールの名前と型を表示
        module_type = module.__class__.__name__
        print(f"{prefix}{'└─' if prefix else ''} {path} ({module_type})")
        
        # このモジュールのパラメータ数をカウント
        module_params = 0
        module_trainable_params = 0
        
        for name, param in module.named_parameters(recurse=False):
            param_count = param.numel()
            module_params += param_count
            if param.requires_grad:
                module_trainable_params += param_count
            
            # 全体のカウントも更新
            total_params += param_count
            if param.requires_grad:
                total_trainable_params += param_count
            
            # 詳細情報を記録
            full_param_name = f"{path}.{name}" if path else name
            param_details[full_param_name] = {
                "shape": list(param.shape),
                "numel": param.numel(),
                "requires_grad": param.requires_grad,
                "module_type": module_type
            }
            
            # パラメータの詳細を表示（最大深度の場合のみ）
            if depth == max_depth:
                requires_grad_str = "✓" if param.requires_grad else "✗"
                print(f"{prefix}   ├─ {name}: {param.shape}, params={param.numel():,}, requires_grad={requires_grad_str}")
        
        # このモジュールのパラメータ情報を表示
        if module_params > 0:
            percent_trainable = (module_trainable_params / module_params) * 100 if module_params > 0 else 0
            print(f"{prefix}   ├─ パラメータ: {module_params:,} (学習可能: {module_trainable_params:,}, {percent_trainable:.1f}%)")
        
        # サブモジュールを再帰的に処理
        child_prefix = prefix + "   "
        for i, (name, child) in enumerate(module.named_children()):
            is_last = i == len(list(module.named_children())) - 1
            new_prefix = child_prefix + ("└─ " if is_last else "├─ ")
            new_path = f"{path}.{name}" if path else name
            _print_module(child, child_prefix, depth + 1, new_path)
    
    # トップレベルのモジュールから開始
    _print_module(model)
    
    # 学習可能/不可能なパラメータの合計を表示
    print(f"\n===== パラメータ統計 =====")
    print(f"総パラメータ数: {total_params:,}")
    print(f"学習可能なパラメータ数: {total_trainable_params:,} ({total_trainable_params/total_params*100:.2f}%)")
    
    # 主要コンポーネント別の学習可能パラメータの割合を表示
    component_stats = {}
    for full_name, details in param_details.items():
        # コンポーネント名の抽出（最初のドットまで）
        component = full_name.split('.')[0] if '.' in full_name else 'other'
        
        if component not in component_stats:
            component_stats[component] = {
                "total": 0,
                "trainable": 0
            }
        
        component_stats[component]["total"] += details["numel"]
        if details["requires_grad"]:
            component_stats[component]["trainable"] += details["numel"]
    
    # コンポーネント別の統計を表示
    print("\n===== コンポーネント別統計 =====")
    for component, stats in sorted(component_stats.items(), key=lambda x: x[1]["total"], reverse=True):
        percent = (stats["trainable"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"{component}: 合計={stats['total']:,}, 学習可能={stats['trainable']:,} ({percent:.1f}%)")
    
    print("=========================================================\n")


# SAM2モデルにLoRAを適用するための設定
def create_lora_sam2(model):
    """SAM2モデルにLoRAを適用し、重要なレイヤーを学習可能に設定"""
    print("SAM2モデルの初期化と学習可能なパラメータの設定を開始します...")
    
    # 1. まずモデル全体の構造を分析
    print_model_structure(model, max_depth=3)
    
    # 2. デフォルトでは全パラメータを凍結
    for param in model.parameters():
        param.requires_grad = False
    
    # 3. 学習対象とするコンポーネントを特定して解凍
    target_modules = [
        "mask_decoder",         # マスクデコーダー全体
        "memory_encoder",       # メモリエンコーダー
        "prompt_encoder",       # プロンプトエンコーダー
        "transformer",          # Transformerモジュール
        "mlp"                   # MLPレイヤー
    ]
    
    # 4. 指定したモジュールとその子モジュールを学習可能に設定
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            for param_name, param in module.named_parameters(recurse=False):
                param.requires_grad = True
                print(f"学習可能に設定: {name}.{param_name}")
    
    # 5. さらに重要なパラメータを個別に学習可能に設定
    key_param_keywords = [
        "decoder.transformer", 
        "decoder.mlp", 
        "memory_encoder", 
        "attention", 
        "transformer.layers",
        "output_upscaling",
        "output_hypernetworks"
    ]
    
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in key_param_keywords):
            param.requires_grad = True
            print(f"追加で学習可能に設定: {name}")
    
    # 6. 画像エンコーダは基本的に凍結する
    if hasattr(model, "image_encoder"):
        # 画像エンコーダの最後の層だけを学習可能に設定（オプション）
        try:
            for name, param in model.image_encoder.named_parameters():
                if 'layer4' in name or 'last_layer' in name:
                    param.requires_grad = True
                    print(f"画像エンコーダの最後の層を学習可能に設定: {name}")
        except AttributeError:
            print("画像エンコーダの構造が想定と異なります。一部の層が見つかりません。")
    
    # 7. 学習可能パラメータの合計を確認
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"学習可能なパラメータ数: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    if trainable_params == 0:
        print("警告: 学習可能なパラメータがありません。モデル構造を確認してください。")
        # 緊急措置として、一部のパラメータを強制的に学習可能に設定
        for name, param in model.named_parameters():
            if "decoder" in name:
                param.requires_grad = True
                print(f"緊急措置: {name} を学習可能に設定")
        
        # 再度確認
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"調整後の学習可能なパラメータ数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # 8. LoRA設定の適用 (PEFT ライブラリを使用)
    try:
        # LoRA設定
        lora_config = LoraConfig(
            r=16,                     # LoRAのランク
            lora_alpha=32,            # スケーリング係数
            target_modules=["q_proj", "v_proj", "out_proj", "fc1", "fc2"],  # ターゲットモジュール
            lora_dropout=0.1,         # LoRAドロップアウト率
            bias="none",              # バイアスの扱い
            task_type=TaskType.TOKEN_CLS  # タスクタイプ
        )
        
        # LoRAモデルの作成
        model = get_peft_model(model, lora_config)
        print("LoRA設定を適用しました。")
        
        # 学習可能パラメータ情報の更新
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"LoRA適用後の学習可能なパラメータ数: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    except Exception as e:
        print(f"LoRA適用中にエラーが発生しました: {e}")
        print("通常のファインチューニングを続行します。")
    
    return model


# 予測マスクを保存して視覚的に検証するための関数
def save_predicted_masks(images, gt_masks, pred_masks, epoch, step, out_dir="debug_vis"):
    """
    予測マスクを視覚的に検証するための画像を保存
    
    Args:
        images: 入力画像 (バッチ)
        gt_masks: 正解マスク (バッチ)
        pred_masks: 予測マスク (バッチ)
        epoch: エポック番号
        step: ステップ番号
        out_dir: 出力ディレクトリ
    """
    os.makedirs(out_dir, exist_ok=True)
    
    for i in range(len(images)):
        # テンソルをnumpy配列に変換
        img = images[i].cpu().permute(1, 2, 0).numpy() * 255
        img = img.astype(np.uint8)
        
        # GTマスクをテンソルからnumpy配列に変換し、チャネル次元を適切に処理
        if gt_masks.dim() == 4:  # [batch, channel, height, width]
            gt_mask = gt_masks[i].cpu().squeeze(0).numpy()  # チャネル次元だけを削除
        else:  # [batch, height, width]
            gt_mask = gt_masks[i].cpu().numpy()
            
        print(f"GT マスク変換後のサイズ: {gt_mask.shape}")
        
        # 予測マスクをシグモイドで0-1に変換
        if pred_masks.dim() == 4:  # [batch, channel, height, width]
            pred_mask = torch.sigmoid(pred_masks[i].detach()).cpu().squeeze(0).numpy() > 0.5
        else:  # [batch, height, width]
            pred_mask = torch.sigmoid(pred_masks[i].detach()).cpu().numpy() > 0.5
            
        pred_mask = pred_mask.astype(np.uint8)
        print(f"予測マスク変換後のサイズ: {pred_mask.shape}")
        
        # GTマスクからカラーマップを作成
        if len(gt_mask.shape) == 2:  # 2次元の場合
            gt_mask_colored = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
            gt_mask_colored[gt_mask > 0] = [0, 255, 0]  # 緑色で正解マスクを表示
        else:
            print(f"警告: GTマスクの次元が想定外です: {gt_mask.shape}")
            # 3D以上の場合、最初の2次元を使用
            temp_mask = gt_mask if len(gt_mask.shape) == 2 else gt_mask[:, :, 0]
            gt_mask_colored = np.zeros((temp_mask.shape[0], temp_mask.shape[1], 3), dtype=np.uint8)
            gt_mask_colored[temp_mask > 0] = [0, 255, 0]
        
        # 予測マスクからカラーマップを作成
        if len(pred_mask.shape) == 2:  # 2次元の場合
            pred_mask_colored = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
            pred_mask_colored[pred_mask > 0] = [255, 0, 0]  # 赤色で予測マスクを表示
        else:
            print(f"警告: 予測マスクの次元が想定外です: {pred_mask.shape}")
            # 3D以上の場合、最初の2次元を使用
            temp_mask = pred_mask if len(pred_mask.shape) == 2 else pred_mask[:, :, 0]
            pred_mask_colored = np.zeros((temp_mask.shape[0], temp_mask.shape[1], 3), dtype=np.uint8)
            pred_mask_colored[temp_mask > 0] = [255, 0, 0]
        
        # デバッグ情報: 各画像のサイズを出力
        print(f"画像サイズ: {img.shape}, GT マスクサイズ: {gt_mask_colored.shape}, 予測マスクサイズ: {pred_mask_colored.shape}")
        
        # マスクのサイズが画像と一致しない場合はリサイズ
        if img.shape[:2] != gt_mask_colored.shape[:2]:
            print(f"GTマスクのリサイズが必要: {gt_mask_colored.shape[:2]} -> {img.shape[:2]}")
            gt_mask_colored = cv2.resize(gt_mask_colored, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        if img.shape[:2] != pred_mask_colored.shape[:2]:
            print(f"予測マスクのリサイズが必要: {pred_mask_colored.shape[:2]} -> {img.shape[:2]}")
            pred_mask_colored = cv2.resize(pred_mask_colored, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # 透明度を設定して元画像に重ねる
        alpha = 0.5
        gt_overlay = cv2.addWeighted(img, 1, gt_mask_colored, alpha, 0)
        pred_overlay = cv2.addWeighted(img, 1, pred_mask_colored, alpha, 0)
        
        # 全ての画像が同じサイズであることを確認
        print(f"連結前の画像サイズ - 元画像: {img.shape}, GT重ね: {gt_overlay.shape}, 予測重ね: {pred_overlay.shape}")
        
        # サイズが異なる場合は共通サイズにリサイズ
        if not (img.shape == gt_overlay.shape == pred_overlay.shape):
            print("連結前に画像サイズを統一します")
            # 最小の高さと幅を計算
            min_height = min(img.shape[0], gt_overlay.shape[0], pred_overlay.shape[0])
            min_width = min(img.shape[1], gt_overlay.shape[1], pred_overlay.shape[1])
            
            # 全ての画像を同じサイズにリサイズ
            img = cv2.resize(img, (min_width, min_height))
            gt_overlay = cv2.resize(gt_overlay, (min_width, min_height))
            pred_overlay = cv2.resize(pred_overlay, (min_width, min_height))
        
        # 比較画像を水平に連結
        try:
            comparison = np.concatenate([
                img, 
                gt_overlay,
                pred_overlay
            ], axis=1)
            
            # 画像の保存
            output_path = os.path.join(out_dir, f"epoch_{epoch}_step_{step}_sample_{i}.png")
            cv2.imwrite(output_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"画像連結中にエラーが発生しました: {e}")
            print(f"画像サイズ: {img.shape}, GT重ね: {gt_overlay.shape}, 予測重ね: {pred_overlay.shape}")


# カスタムcollate関数の定義（グローバルスコープ）
def custom_collate_fn(batch):
    """
    カスタムのcollate関数
    バッチ内のマスクのサイズが異なる問題を解決
    """
    # バッチからデータを抽出
    images = [item['image'] for item in batch]
    prompt_points = [item['prompt_points'] for item in batch]
    prompt_labels = [item['prompt_labels'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]
    input_sizes = [item['input_size'] for item in batch]
    
    # マスクはカテゴリごとにまとめる
    masks_dict = {}
    for item in batch:
        for cat_id, mask in item['masks'].items():
            if cat_id not in masks_dict:
                masks_dict[cat_id] = []
            masks_dict[cat_id].append(mask)
    
    # 同じサイズのテンソルをスタックする
    images = torch.stack(images)
    
    # 可変長のテンソルはそのままリストで渡す
    # プロンプトのポイントとラベルは可変長のため、リストのまま
    
    # collateした結果を辞書にまとめる
    collated_batch = {
        'image': images,
        'masks': {cat_id: torch.stack(masks) for cat_id, masks in masks_dict.items()},
        'prompt_points': prompt_points,
        'prompt_labels': prompt_labels,
        'image_id': image_ids,
        'original_size': original_sizes,
        'input_size': input_sizes
    }
    
    return collated_batch


# トレーニングループ
def train_sam2_for_manga(model, train_loader, val_loader, device, num_epochs=10, log_dir="logs"):
    """
    SAM2モデルをトレーニングするメイン関数
    モデルのパラメータが適切に学習されるように最適化設定を調整
    
    Args:
        model: ファインチューニングするSAM2モデル
        train_loader: トレーニングデータローダー
        val_loader: 検証データローダー
        device: 計算デバイス
        num_epochs: エポック数
        log_dir: ログを保存するディレクトリ
        
    Returns:
        トレーニング済みモデル
    """
    print("学習プロセスを初期化しています...")
    
    # ログディレクトリの作成
    os.makedirs(log_dir, exist_ok=True)
    
    # 現在の日時を取得してログファイル名を生成
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"sam2_finetune_{timestamp}.log")
    
    # ロガーのセットアップ
    logger = setup_logger(log_file)
    logger.info(f"SAM2ファインチューニング開始: {timestamp}")
    logger.info(f"デバイス: {device}")
    logger.info(f"ログファイル: {log_file}")
    
    # データセット情報のログ記録
    logger.info(f"トレーニングデータセットサイズ: {len(train_loader.dataset)}枚")
    logger.info(f"検証データセットサイズ: {len(val_loader.dataset)}枚")
    logger.info(f"バッチサイズ: {train_loader.batch_size}")
    logger.info(f"エポック数: {num_epochs}")
    
    model = model.to(device)
    
    # 学習可能なパラメータが存在するか確認
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        logger.warning("重大な警告: モデル内に学習可能なパラメータが見つかりません。トレーニングを正しく行うために対応します。")
        # 一部パラメータを学習可能に設定（マスクデコーダーとプロンプトエンコーダーを優先）
        for name, param in model.named_parameters():
            if "mask_decoder" in name or "prompt_encoder" in name or "memory_encoder" in name:
                param.requires_grad = True
                logger.info(f"パラメータを強制的に学習可能に設定: {name}")
    
    # 学習率の異なるパラメータグループを設定
    param_groups = [
        # デコーダー部分は高い学習率で
        {'params': [p for n, p in model.named_parameters() if 'decoder' in n and p.requires_grad], 'lr': 1e-4},
        # エンコーダー部分は低い学習率で（まだ残っている学習可能なものがあれば）
        {'params': [p for n, p in model.named_parameters() if 'encoder' in n and p.requires_grad], 'lr': 5e-5},
        # その他の学習可能パラメータ
        {'params': [p for n, p in model.named_parameters() if not ('encoder' in n or 'decoder' in n) and p.requires_grad], 'lr': 8e-5}
    ]
    
    # 各パラメータグループの要素数を表示
    for i, group in enumerate(param_groups):
        logger.info(f"パラメータグループ {i+1}: {len(group['params'])} パラメータ, lr={group['lr']}")
    
    # AdamWオプティマイザでの重み減衰と学習率設定
    optimizer = optim.AdamW(
        param_groups,
        weight_decay=0.01,  # 重み減衰を適用
        eps=1e-8  # 数値安定性のため
    )
    
    # 学習率スケジューラの設定
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,  # 学習率を半分に
        patience=2,   # 2エポック改善がなければ減衰
        verbose=True  # 学習率変更時に表示
    )
    
    # 損失関数
    criterion = SegmentationLoss(alpha=0.6)  # ディスロスとIoU損失のバランスを調整
    
    # 損失の履歴を保存するリスト
    train_losses = []
    val_losses = []
    epochs = []
    
    # 再度確認
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"トレーニング開始前の確認:")
    logger.info(f"- 学習可能なパラメータ数: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    logger.info(f"- 使用デバイス: {device}")
    logger.info(f"- バッチサイズ: {train_loader.batch_size}")
    logger.info(f"- エポック数: {num_epochs}")
    
    best_val_loss = float('inf')
    
    # 結果を保存するディレクトリ
    os.makedirs("debug_vis", exist_ok=True)
    
    for epoch in range(num_epochs):
        # エポック開始時間
        epoch_start_time = datetime.datetime.now()
        logger.info(f"エポック {epoch+1}/{num_epochs} 開始: {epoch_start_time.strftime('%H:%M:%S')}")
        
        # トレーニングフェーズ
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device)
            masks = {k: v.to(device) for k, v in batch['masks'].items()}
            
            # プロンプトポイントとラベルはリスト形式に変更されている
            points = [p.to(device) for p in batch['prompt_points']]
            labels = [l.to(device) for l in batch['prompt_labels']]
            
            # バッチ内の画像が空でないことを確認
            if images.shape[0] == 0:
                continue
                
            # 入力画像の前処理
            try:
                image_embeddings = model.image_encoder(images)
            except Exception as e:
                print(f"画像エンコーダでエラーが発生しました: {e}")
                print(f"画像サイズ: {images.shape}")
                # エラーの詳細情報を出力
                traceback.print_exc()
                continue
            
            loss_batch = 0
            # 各カテゴリーについて処理
            for cat_id, gt_mask in masks.items():
                if not torch.any(gt_mask):
                    continue
                
                # 点プロンプトが存在するか確認
                if len(points) > 0 and any(p.shape[0] > 0 for p in points):
                    try:
                        # バッチ内の各サンプルのポイント座標とラベルを準備
                        coords_list = []
                        labels_list = []
                        
                        for i in range(len(points)):  # バッチ内の各サンプルについて
                            sample_points = points[i]
                            sample_labels = labels[i]
                            
                            # 有効なポイント（座標が0でない）のみを使用
                            if sample_points.shape[0] > 0:
                                valid_points = sample_points[torch.any(sample_points != 0, dim=1)] if sample_points.shape[0] > 0 else sample_points
                                if valid_points.shape[0] > 0:
                                    valid_labels = sample_labels[:valid_points.shape[0]] if sample_labels.shape[0] > 0 else torch.ones(valid_points.shape[0], dtype=torch.int, device=device)
                                    coords_list.append(valid_points)
                                    labels_list.append(valid_labels)
                                else:
                                    # 有効なポイントがない場合、ダミーデータを追加
                                    coords_list.append(torch.zeros((1, 2), device=device))
                                    labels_list.append(torch.ones(1, dtype=torch.int, device=device))
                            else:
                                # ポイントがない場合、ダミーデータを追加
                                coords_list.append(torch.zeros((1, 2), device=device))
                                labels_list.append(torch.ones(1, dtype=torch.int, device=device))
                        
                        # 1. 画像エンベディングを取得
                        # image_embeddings はすでに取得済み
                        
                        # 2. プロンプト処理 (ポイントプロンプト)
                        # 最初のサンプルのみを使用（簡略化のため）
                        if coords_list:
                            point_coords = coords_list[0].unsqueeze(0)  # [1, N, 2]
                            point_labels = labels_list[0].unsqueeze(0)  # [1, N]
                            
                            # 画像サイズ情報を取得
                            batch_size, _, height, width = images.shape
                            orig_size = (height, width)
                            
                            try:
                                # プロンプトエンコーダでプロンプト処理
                                if hasattr(model, "prompt_encoder") and hasattr(model.prompt_encoder, "encode_points"):
                                    sparse_embeddings = model.prompt_encoder.encode_points(
                                        point_coords,
                                        point_labels,
                                        orig_size
                                    )
                                else:
                                    # モデル構造が異なる場合のフォールバック
                                    sparse_embeddings = torch.zeros(1, 256, device=device)  # ダミーエンベディング
                            
                                # 3. マスク予測
                                if hasattr(model, "mask_decoder"):
                                    # SAM2の標準構造
                                    if isinstance(image_embeddings, dict) and 'vision_features' in image_embeddings:
                                        vision_features = image_embeddings['vision_features']
                                        mask_predictions = model.mask_decoder(
                                            image_embeddings=vision_features,
                                            prompt_embeddings=sparse_embeddings,
                                        )
                                    else:
                                        # 標準的なデコーダー呼び出し
                                        mask_predictions = model.mask_decoder(
                                            image_embeddings=image_embeddings,
                                            prompt_embeddings=sparse_embeddings,
                                        )
                                elif hasattr(model, "sam_mask_decoder"):
                                    # SAM2の代替構造
                                    try:
                                        # 画像の位置エンコーディングを取得
                                        if hasattr(model, "prompt_encoder") and hasattr(model.prompt_encoder, "get_dense_pe"):
                                            image_pe = model.prompt_encoder.get_dense_pe()
                                        else:
                                            # 画像エンベディングと同じ空間サイズで位置エンコーディングを作成
                                            if isinstance(image_embeddings, torch.Tensor):
                                                h, w = image_embeddings.shape[-2:]
                                                image_pe = torch.zeros((1, 256, h, w), device=device)
                                            else:
                                                # 辞書型の場合は適切なキーからテンソルを取得
                                                if 'vision_features' in image_embeddings:
                                                    vision_features = image_embeddings['vision_features']
                                                    h, w = vision_features.shape[-2:]
                                                    image_pe = torch.zeros((1, 256, h, w), device=device)
                                                else:
                                                    print("警告: 画像埋め込みの形式が不明です。仮の値を使用します。")
                                                    image_pe = torch.zeros((1, 256, 64, 64), device=device)
                                        
                                        # image_embeddingsが辞書型の場合、適切なキーから特徴量を取得
                                        if isinstance(image_embeddings, dict):
                                            if 'vision_features' in image_embeddings:
                                                image_embeddings_tensor = image_embeddings['vision_features']
                                            elif 'encoder_embedding' in image_embeddings:
                                                image_embeddings_tensor = image_embeddings['encoder_embedding']
                                            else:
                                                # キーが見つからない場合はキーの一覧を表示
                                                print(f"利用可能なキー: {list(image_embeddings.keys())}")
                                                raise ValueError("適切な画像特徴量キーが見つかりません")
                                        else:
                                            # すでにテンソルの場合はそのまま使用
                                            image_embeddings_tensor = image_embeddings
                                        
                                        # 画像特徴量のチャネル数とサイズを取得して適切なdense_embeddingsを作成
                                        b, c, h, w = image_embeddings_tensor.shape
                                        print(f"画像特徴量の形状: {image_embeddings_tensor.shape}")
                                        
                                        # 空の dense embeddings を作成（画像特徴量と同じ形状）
                                        # 重要: チャネル数はimage_embeddings_tensorと一致する必要がある
                                        dense_embeddings = torch.zeros((b, c, h, w), device=device)
                                        print(f"dense_embeddingsの形状: {dense_embeddings.shape}")
                                        
                                        # sparse_embeddingsの次元を調整（2次元から3次元に）
                                        # SAM2のマスクデコーダは [batch, tokens, dim] の形状を期待している
                                        try:
                                            print(f"プロンプトエンコーダの出力形状: {sparse_embeddings.shape}")
                                            # モデルが期待する形式に変換
                                            if sparse_embeddings.dim() == 2:
                                                print(f"2次元のsparse_embeddingsを3次元に変換します: {sparse_embeddings.shape} -> ", end='')
                                                sparse_embeddings = sparse_embeddings.unsqueeze(1)
                                                print(f"{sparse_embeddings.shape}")
                                            elif sparse_embeddings.dim() == 1:
                                                print(f"1次元のsparse_embeddingsを3次元に変換します: {sparse_embeddings.shape} -> ", end='')
                                                sparse_embeddings = sparse_embeddings.unsqueeze(0).unsqueeze(0)
                                                print(f"{sparse_embeddings.shape}")
                                            
                                            # バッチサイズ、トークン数、埋め込み次元の確認
                                            if sparse_embeddings.dim() == 3:
                                                b, t, d = sparse_embeddings.shape
                                                print(f"sparse_embeddings: バッチサイズ={b}, トークン数={t}, 次元={d}")
                                            else:
                                                print(f"警告: 3次元でないsparse_embeddings。強制的に3次元に変換します: {sparse_embeddings.shape}")
                                                # 次元がまだ3でない場合は強制的に3次元に変換
                                                if sparse_embeddings.dim() == 1:
                                                    sparse_embeddings = sparse_embeddings.unsqueeze(0).unsqueeze(0)
                                                elif sparse_embeddings.dim() == 2:
                                                    sparse_embeddings = sparse_embeddings.unsqueeze(1)
                                                print(f"変換後: {sparse_embeddings.shape}")
                                        except Exception as e:
                                            print(f"sparse_embeddings処理中のエラー: {e}")
                                            traceback.print_exc()
                                            # 回避策: ダミー埋め込みを作成
                                            sparse_embeddings = torch.zeros((1, 1, 256), device=device)
                                        
                                        # バッチサイズの一致を確認し、必要に応じて調整
                                        img_batch_size = image_embeddings_tensor.shape[0]
                                        prompt_batch_size = sparse_embeddings.shape[0]
                                        
                                        if img_batch_size != prompt_batch_size:
                                            print(f"バッチサイズ不一致を調整します: 画像={img_batch_size}, プロンプト={prompt_batch_size}")
                                            if img_batch_size > prompt_batch_size:
                                                # sparse_embeddingsを複製して同じバッチサイズにする
                                                sparse_embeddings = sparse_embeddings.repeat(img_batch_size, 1, 1)
                                                # dense_embeddingsはすでに正しいバッチサイズを持っているので、調整は不要
                                            else:
                                                # 先頭のバッチだけを使用
                                                image_embeddings_tensor = image_embeddings_tensor[:prompt_batch_size]
                                                dense_embeddings = dense_embeddings[:prompt_batch_size]
                                                image_pe = image_pe[:prompt_batch_size] if image_pe.dim() > 3 else image_pe

                                        # 高解像度特徴がある場合はそれを使用
                                        high_res_features = None
                                        # use_high_res_features_in_sam フラグがモデルで設定されている場合、
                                        # 空のリストを渡すのではなくNoneを渡す
                                        
                                        # 正しい順序で引数を渡してマスクデコーダを呼び出し
                                        mask_predictions = model.sam_mask_decoder(
                                            image_embeddings_tensor,  # 画像の埋め込み（テンソル形式に変換済み）
                                            image_pe,               # 画像の位置エンコーディング
                                            sparse_embeddings,      # スパースプロンプトの埋め込み（3次元に調整済み）
                                            dense_embeddings,       # 密なプロンプトの埋め込み
                                            False,                  # multimask_output: 単一マスク出力
                                            False,                  # repeat_image: 画像の繰り返しなし
                                            high_res_features       # high_res_features: Noneを明示的に使用
                                        )
                                        
                                        # マスク予測が複数出力の場合、最初の要素を使用
                                        if isinstance(mask_predictions, tuple):
                                            print("マスクデコーダの出力はタプル形式です。最初の要素（マスク）を使用します。")
                                            mask_predictions = mask_predictions[0]
                                    except Exception as e:
                                        print(f"マスクデコーダの呼び出し中にエラー発生: {e}")
                                        traceback.print_exc()  # 詳細なエラー情報を表示
                                        # フォールバック
                                        mask_predictions = torch.zeros((batch_size, 1, height, width), device=device)
                                        mask_predictions.requires_grad_(True)
                                else:
                                    # フォールバック
                                    mask_predictions = torch.zeros((batch_size, 1, height, width), device=device)
                                    mask_predictions.requires_grad_(True)
                                    
                                # マスク予測が複数出力の場合、最初の要素を使用
                                if isinstance(mask_predictions, tuple):
                                    mask_predictions = mask_predictions[0]
                                    
                            except Exception as e:
                                print(f"マスク予測中にエラーが発生しました: {e}")
                                traceback.print_exc()
                                mask_predictions = torch.zeros((batch_size, 1, height, width), device=device)
                                mask_predictions.requires_grad_(True)
                    except Exception as e:
                        print(f"プロンプト処理中にエラーが発生しました: {e}")
                        traceback.print_exc()
                        mask_predictions = torch.zeros((batch_size, 1, height, width), device=device)
                        mask_predictions.requires_grad_(True)
                else:
                    # プロンプトがない場合はダミーのマスク予測を生成
                    print("プロンプトが存在しないためスキップします")
                    continue
                
                # GTマスクの形状調整（必要に応じて）
                if len(gt_mask.shape) == 3:  # [バッチ, 高さ, 幅]
                    gt_mask = gt_mask.unsqueeze(1)  # [バッチ, 1, 高さ, 幅]
                elif len(gt_mask.shape) == 2:  # [高さ, 幅]
                    gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 高さ, 幅]
                
                # マスク予測と教師マスクのサイズが異なる場合は調整
                if mask_predictions.shape != gt_mask.shape:
                    gt_mask = torch.nn.functional.interpolate(
                        gt_mask.float(), 
                        size=mask_predictions.shape[2:], 
                        mode='nearest'
                    )
                
                # 損失計算
                try:
                    # 勾配計算のための確認
                    if not mask_predictions.requires_grad:
                        mask_predictions.requires_grad_(True)
                    
                    loss = criterion(mask_predictions, gt_mask.float())
                    
                    # NaNやInf値のチェック
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"警告: 損失値が無効です: {loss.item()}")
                        loss = 0.1 * torch.mean(mask_predictions**2)  # 代替損失
                except Exception as e:
                    print(f"損失計算でエラー: {e}")
                    # 代替損失を使用
                    dummy_param = next((p for p in model.parameters() if p.requires_grad), None)
                    if dummy_param is not None:
                        loss = 0.1 * torch.mean(dummy_param**2)
                    else:
                        loss = torch.tensor(0.1, device=device, requires_grad=True)
                
                loss_batch += loss
                
                # 一定間隔で予測マスクを視覚化して保存
                if batch_idx % 10 == 0:
                    save_predicted_masks(
                        images, 
                        gt_mask, 
                        mask_predictions, 
                        epoch, 
                        batch_idx,
                        out_dir="debug_vis"
                    )
            
            # 逆伝播と最適化
            optimizer.zero_grad()
            loss_batch.backward()
            
            # グラディエントクリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss_batch.item()
            progress_bar.set_postfix({'loss': loss_batch.item()})
            
            # 一定間隔でバッチ情報をログに記録
            if batch_idx % 20 == 0:
                logger.info(f"エポック {epoch+1}, バッチ {batch_idx}/{len(train_loader)}, 損失: {loss_batch.item():.4f}")
        
        train_loss /= len(train_loader)
        logger.info(f"エポック {epoch+1} トレーニング完了 - 平均損失: {train_loss:.4f}")
        
        # バリデーションフェーズ
        logger.info(f"エポック {epoch+1} 検証フェーズ開始")
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validation')):
                images = batch['image'].to(device)
                masks = {k: v.to(device) for k, v in batch['masks'].items()}
                points = [p.to(device) for p in batch['prompt_points']]
                labels = [l.to(device) for l in batch['prompt_labels']]
                
                # 画像エンベディングを取得
                image_embeddings = model.image_encoder(images)
                
                loss_batch = 0
                for cat_id, gt_mask in masks.items():
                    if not torch.any(gt_mask):
                        continue
                    
                    # 点プロンプトが存在する場合のみ処理
                    if len(points) > 0 and any(p.shape[0] > 0 for p in points):
                        # バッチの最初のサンプルを使用（簡易化）
                        sample_points = points[0]
                        sample_labels = labels[0]
                        
                        if sample_points.shape[0] > 0:
                            valid_points = sample_points[torch.any(sample_points != 0, dim=1)] if sample_points.shape[0] > 0 else sample_points
                            if valid_points.shape[0] > 0:
                                valid_labels = sample_labels[:valid_points.shape[0]] if sample_labels.shape[0] > 0 else torch.ones(valid_points.shape[0], dtype=torch.int, device=device)
                                point_coords = valid_points.unsqueeze(0)
                                point_labels = valid_labels.unsqueeze(0)
                            else:
                                point_coords = torch.zeros((1, 1, 2), device=device)
                                point_labels = torch.ones(1, 1, dtype=torch.int, device=device)
                        else:
                            point_coords = torch.zeros((1, 1, 2), device=device)
                            point_labels = torch.ones(1, 1, dtype=torch.int, device=device)
                        
                        batch_size, _, height, width = images.shape
                        orig_size = (height, width)
                        
                        # プロンプト処理
                        if hasattr(model, "prompt_encoder") and hasattr(model.prompt_encoder, "encode_points"):
                            sparse_embeddings = model.prompt_encoder.encode_points(
                                point_coords,
                                point_labels,
                                orig_size
                            )
                            
                            # トレーニングフェーズでもsparse_embeddingsの次元を調整
                            if sparse_embeddings.dim() == 2:
                                print(f"トレーニング時: 2次元のsparse_embeddingsを3次元に変換します: {sparse_embeddings.shape} -> ", end='')
                                sparse_embeddings = sparse_embeddings.unsqueeze(1)
                                print(f"{sparse_embeddings.shape}")
                            elif sparse_embeddings.dim() == 1:
                                print(f"トレーニング時: 1次元のsparse_embeddingsを3次元に変換します: {sparse_embeddings.shape} -> ", end='')
                                sparse_embeddings = sparse_embeddings.unsqueeze(0).unsqueeze(0)
                                print(f"{sparse_embeddings.shape}")
                        else:
                            sparse_embeddings = torch.zeros(1, 256, device=device)
                        
                        # マスク予測
                        if hasattr(model, "mask_decoder"):
                            if isinstance(image_embeddings, dict) and 'vision_features' in image_embeddings:
                                vision_features = image_embeddings['vision_features']
                                mask_predictions = model.mask_decoder(
                                    image_embeddings=vision_features,
                                    prompt_embeddings=sparse_embeddings,
                                )
                            else:
                                mask_predictions = model.mask_decoder(
                                    image_embeddings=image_embeddings,
                                    prompt_embeddings=sparse_embeddings,
                                )
                        elif hasattr(model, "sam_mask_decoder"):
                            # SAM2のマスクデコーダは異なるインターフェースを使用
                            # 必要な引数: image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output, repeat_image
                            
                            # 画像の位置エンコーディングを取得（モデルから取得できない場合はゼロテンソルを使用）
                            if hasattr(model, "prompt_encoder") and hasattr(model.prompt_encoder, "get_dense_pe"):
                                image_pe = model.prompt_encoder.get_dense_pe()
                            else:
                                # 画像エンベディングと同じ空間サイズで位置エンコーディングを作成
                                if isinstance(image_embeddings, torch.Tensor):
                                    h, w = image_embeddings.shape[-2:]
                                    image_pe = torch.zeros((1, 256, h, w), device=device)
                                else:
                                    # ディクショナリの場合は仮の値を設定
                                    image_pe = torch.zeros((1, 256, 64, 64), device=device)
                            
                            # image_embeddingsが辞書型の場合、適切なキーから特徴量を取得
                            if isinstance(image_embeddings, dict):
                                if 'vision_features' in image_embeddings:
                                    image_embeddings_tensor = image_embeddings['vision_features']
                                elif 'encoder_embedding' in image_embeddings:
                                    image_embeddings_tensor = image_embeddings['encoder_embedding']
                                else:
                                    # キーが見つからない場合はキーの一覧を表示
                                    print(f"検証時の利用可能なキー: {list(image_embeddings.keys())}")
                                    raise ValueError("検証時に適切な画像特徴量キーが見つかりません")
                            else:
                                # すでにテンソルの場合はそのまま使用
                                image_embeddings_tensor = image_embeddings
                            
                            # 画像特徴量のチャネル数とサイズを取得して適切なdense_embeddingsを作成
                            b, c, h, w = image_embeddings_tensor.shape
                            print(f"検証時の画像特徴量の形状: {image_embeddings_tensor.shape}")
                            
                            # 空の dense embeddings を作成（画像特徴量と同じ形状）
                            # 重要: チャネル数はimage_embeddings_tensorと一致する必要がある
                            dense_embeddings = torch.zeros((b, c, h, w), device=device)
                            print(f"検証時のdense_embeddingsの形状: {dense_embeddings.shape}")
                                 # sparse_embeddingsの次元を調整（2次元から3次元に）
                        # SAM2のマスクデコーダは [batch, tokens, dim] の形状を期待している
                        try:
                            print(f"検証時のプロンプトエンコーダの出力形状: {sparse_embeddings.shape}")
                            # モデルが期待する形式に変換
                            if sparse_embeddings.dim() == 2:
                                print(f"検証時: 2次元のsparse_embeddingsを3次元に変換します: {sparse_embeddings.shape} -> ", end='')
                                sparse_embeddings = sparse_embeddings.unsqueeze(1)
                                print(f"{sparse_embeddings.shape}")
                            elif sparse_embeddings.dim() == 1:
                                print(f"検証時: 1次元のsparse_embeddingsを3次元に変換します: {sparse_embeddings.shape} -> ", end='')
                                sparse_embeddings = sparse_embeddings.unsqueeze(0).unsqueeze(0)
                                print(f"{sparse_embeddings.shape}")
                            
                            # バッチサイズ、トークン数、埋め込み次元の確認
                            if sparse_embeddings.dim() == 3:
                                b, t, d = sparse_embeddings.shape
                                print(f"検証時のsparse_embeddings: バッチサイズ={b}, トークン数={t}, 次元={d}")
                            else:
                                print(f"検証時の警告: 3次元でないsparse_embeddings。強制的に3次元に変換します: {sparse_embeddings.shape}")
                                # 次元がまだ3でない場合は強制的に3次元に変換
                                if sparse_embeddings.dim() == 1:
                                    sparse_embeddings = sparse_embeddings.unsqueeze(0).unsqueeze(0)
                                elif sparse_embeddings.dim() == 2:
                                    sparse_embeddings = sparse_embeddings.unsqueeze(1)
                                print(f"変換後: {sparse_embeddings.shape}")
                        except Exception as e:
                            print(f"検証時のsparse_embeddings処理中のエラー: {e}")
                            traceback.print_exc()
                            # 回避策: ダミー埋め込みを作成
                            sparse_embeddings = torch.zeros((1, 1, 256), device=device)
                        
                        # バッチサイズの一致を確認し、必要に応じて調整
                        img_batch_size = image_embeddings_tensor.shape[0]
                        prompt_batch_size = sparse_embeddings.shape[0]
                        
                        if img_batch_size != prompt_batch_size:
                            print(f"検証時のバッチサイズ不一致を調整: 画像={img_batch_size}, プロンプト={prompt_batch_size}")
                            if img_batch_size > prompt_batch_size:
                                # sparse_embeddingsを複製して同じバッチサイズにする
                                sparse_embeddings = sparse_embeddings.repeat(img_batch_size, 1, 1)
                                # dense_embeddingsはすでに正しいバッチサイズを持っているので、調整は不要
                            else:
                                # 先頭のバッチだけを使用
                                image_embeddings_tensor = image_embeddings_tensor[:prompt_batch_size]
                                dense_embeddings = dense_embeddings[:prompt_batch_size]
                                image_pe = image_pe[:prompt_batch_size] if image_pe.dim() > 3 else image_pe
                        
                        try:
                            # スパース埋め込みの次元チェックと調整
                            print(f"検証時のスパース埋め込みの形状（調整前）: {sparse_embeddings.shape}")
                            if sparse_embeddings.dim() == 2:
                                sparse_embeddings = sparse_embeddings.unsqueeze(1)  # [B, D] -> [B, 1, D]
                                print(f"検証時のスパース埋め込みの形状（調整後）: {sparse_embeddings.shape}")
                            elif sparse_embeddings.dim() == 1:
                                sparse_embeddings = sparse_embeddings.unsqueeze(0).unsqueeze(0)  # [D] -> [1, 1, D]
                                print(f"検証時のスパース埋め込みの形状（調整後）: {sparse_embeddings.shape}")
                            
                            # マスクデコーダの呼び出し                                
                            mask_predictions = model.sam_mask_decoder(
                                    image_embeddings_tensor,  # 画像の埋め込み（テンソル形式に変換済み）
                                    image_pe,                 # 画像の位置エンコーディング
                                    sparse_embeddings,        # スパースプロンプトの埋め込み（3次元に調整済み）
                                    dense_embeddings,         # 密なプロンプトの埋め込み
                                    False,                    # multimask_output: 単一マスク出力
                                    False,                    # repeat_image: 画像の繰り返しなし
                                    None                      # high_res_features: Noneを明示的に使用
                                )
                            
                            # マスク予測が複数出力の場合、最初の要素を使用
                            if isinstance(mask_predictions, tuple):
                                print("検証時: マスクデコーダの出力はタプル形式です。最初の要素（マスク）を使用します。")
                                mask_predictions = mask_predictions[0]
                        except Exception as e:
                            print(f"検証時のマスクデコーダ呼び出しでエラー: {e}")
                            print(f"画像埋め込み: {image_embeddings_tensor.shape}")
                            print(f"画像PE: {image_pe.shape}")
                            print(f"スパース埋め込み: {sparse_embeddings.shape}")
                            print(f"密な埋め込み: {dense_embeddings.shape}")
                            traceback.print_exc()
                            
                            # エラー回復のために簡易的な出力を生成
                            batch_size, _, height, width = images.shape
                            mask_predictions = torch.zeros((batch_size, 1, height, width), device=device)
                            mask_predictions.requires_grad_(True)
                        else:
                            mask_predictions = torch.zeros((batch_size, 1, height, width), device=device)
                        
                        # タプルの場合は最初の要素を使用
                        if isinstance(mask_predictions, tuple):
                            mask_predictions = mask_predictions[0]
                        
                        # GTマスクの形状調整
                        if len(gt_mask.shape) == 3:
                            gt_mask = gt_mask.unsqueeze(1)
                        elif len(gt_mask.shape) == 2:
                            gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
                        
                        # マスクサイズの調整
                        if mask_predictions.shape != gt_mask.shape:
                            gt_mask = torch.nn.functional.interpolate(
                                gt_mask.float(), 
                                size=mask_predictions.shape[2:], 
                                mode='nearest'
                            )
                        
                        # 損失計算
                        loss = criterion(mask_predictions, gt_mask.float())
                        loss_batch += loss
                        
                        # 検証用に予測マスクを視覚化（サンプルのみ）
                        if batch_idx % 10 == 0 and batch_idx < 50:
                            save_predicted_masks(
                                images, 
                                gt_mask, 
                                mask_predictions, 
                                epoch, 
                                batch_idx + 1000,  # トレーニングバッチと区別するためのオフセット
                                out_dir="debug_vis/val"
                            )
                
                val_loss += loss_batch.item()
        
        val_loss /= len(val_loader)
        
        # エポック終了時間と実行時間の計算
        epoch_end_time = datetime.datetime.now()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # 学習率の調整
        scheduler.step(val_loss)
        
        # 現在の学習率を取得
        current_lr = [group['lr'] for group in optimizer.param_groups]
        
        # エポック結果をログに記録
        logger.info(f'エポック {epoch+1} 完了, トレーニング損失: {train_loss:.4f}, 検証損失: {val_loss:.4f}')
        logger.info(f'エポック実行時間: {epoch_duration}, 現在の学習率: {current_lr}')
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 損失値を記録
        epochs.append(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 最良モデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = 'best_sam2_manga_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, model_save_path)
            logger.info(f"より良い検証損失 ({val_loss:.4f}) を達成したため、モデルを保存しました: {model_save_path}")
    
    # トレーニング終了後に損失の推移をプロット
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # グラフを保存
    plt.savefig('training_history.png')
    plt.show()
    
    return model


# 評価関数
def evaluate_sam2(model, val_loader, device, log_file=None):
    """
    SAM2モデルの評価を行う関数
    
    Args:
        model: 評価するSAM2モデル
        val_loader: 検証データローダー
        device: 計算デバイス
        log_file: ログファイルのパス（オプション）
        
    Returns:
        平均IoUスコア, 平均Diceスコア
    """
    # ロガーのセットアップ
    logger = setup_logger(log_file)
    logger.info("モデル評価プロセスを開始します")
    logger.info(f"評価データセットサイズ: {len(val_loader.dataset)}枚")
    
    # 損失関数
    criterion = SegmentationLoss(alpha=0.6)  # トレーニングと同じ損失関数を使用
    
    model.eval()
    iou_scores = []
    dice_scores = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Evaluating')):
            # 一定間隔でバッチ情報をログに記録
            if batch_idx % 10 == 0:
                logger.info(f"評価バッチ {batch_idx}/{len(val_loader)} 処理中")
                
            images = batch['image'].to(device)
            masks = {k: v.to(device) for k, v in batch['masks'].items()}
            points = [p.to(device) for p in batch['prompt_points']]
            labels = [l.to(device) for l in batch['prompt_labels']]
            
            # 画像エンベディングを取得
            image_embeddings = model.image_encoder(images)
            
            for cat_id, gt_mask in masks.items():
                if not torch.any(gt_mask):
                    continue
                
                # 点プロンプトが存在する場合のみ処理
                if len(points) > 0 and any(p.shape[0] > 0 for p in points):
                    # バッチの最初のサンプルを使用（簡易化）
                    sample_points = points[0]
                    sample_labels = labels[0]
                    
                    if sample_points.shape[0] > 0:
                        valid_points = sample_points[torch.any(sample_points != 0, dim=1)] if sample_points.shape[0] > 0 else sample_points
                        if valid_points.shape[0] > 0:
                            valid_labels = sample_labels[:valid_points.shape[0]] if sample_labels.shape[0] > 0 else torch.ones(valid_points.shape[0], dtype=torch.int, device=device)
                            point_coords = valid_points.unsqueeze(0)
                            point_labels = valid_labels.unsqueeze(0)
                        else:
                            point_coords = torch.zeros((1, 1, 2), device=device)
                            point_labels = torch.ones(1, 1, dtype=torch.int, device=device)
                    else:
                        point_coords = torch.zeros((1, 1, 2), device=device)
                        point_labels = torch.ones(1, 1, dtype=torch.int, device=device)
                    
                    batch_size, _, height, width = images.shape
                    orig_size = (height, width)
                    
                    # プロンプト処理
                    if hasattr(model, "prompt_encoder") and hasattr(model.prompt_encoder, "encode_points"):
                        sparse_embeddings = model.prompt_encoder.encode_points(
                            point_coords,
                            point_labels,
                            orig_size
                        )
                    else:
                        sparse_embeddings = torch.zeros(1, 256, device=device)
                    
                    # マスク予測
                    if hasattr(model, "mask_decoder"):
                        if isinstance(image_embeddings, dict) and 'vision_features' in image_embeddings:
                            vision_features = image_embeddings['vision_features']
                            mask_predictions = model.mask_decoder(
                                image_embeddings=vision_features,
                                prompt_embeddings=sparse_embeddings,
                            )
                        else:
                            mask_predictions = model.mask_decoder(
                                image_embeddings=image_embeddings,
                                prompt_embeddings=sparse_embeddings,
                            )
                    elif hasattr(model, "sam_mask_decoder"):
                        # 画像の位置エンコーディングを取得
                        if hasattr(model, "prompt_encoder") and hasattr(model.prompt_encoder, "get_dense_pe"):
                            image_pe = model.prompt_encoder.get_dense_pe()
                        else:
                            # 画像エンベディングと同じ空間サイズで位置エンコーディングを作成
                            if isinstance(image_embeddings, torch.Tensor):
                                h, w = image_embeddings.shape[-2:]
                                image_pe = torch.zeros((1, 256, h, w), device=device)
                            else:
                                # 辞書型の場合は適切なキーからテンソルを取得
                                if 'vision_features' in image_embeddings:
                                    vision_features = image_embeddings['vision_features']
                                    h, w = vision_features.shape[-2:]
                                    image_pe = torch.zeros((1, 256, h, w), device=device)
                                else:
                                    print("警告: 検証時に画像埋め込みの形式が不明です。仮の値を使用します。")
                                    image_pe = torch.zeros((1, 256, 64, 64), device=device)
                        
                        # 空の dense embeddings を作成
                        dense_embeddings = torch.zeros((1, 1, 256), device=device)
                        
                        # sparse_embeddingsの次元を調整（2次元から3次元に）
                        # SAM2のマスクデコーダは [batch, tokens, dim] の形状を期待している
                        try:
                            print(f"検証時のsparse_embeddings形状: {sparse_embeddings.shape}, 次元数: {sparse_embeddings.dim()}")
                            # モデルが期待する形式に変換
                            if sparse_embeddings.dim() == 2:
                                print(f"2次元のsparse_embeddingsを3次元に変換します: {sparse_embeddings.shape} -> ", end='')
                                # 次元変換を修正：[B, D] -> [B, 1, D] に変換（トークン次元を追加）
                                # 1次元目（バッチ）と2次元目（特徴量）の間に新しい次元を挿入
                                sparse_embeddings = sparse_embeddings.unsqueeze(1)
                                print(f"{sparse_embeddings.shape}")
                            elif sparse_embeddings.dim() == 1:
                                print(f"1次元のsparse_embeddingsを3次元に変換します: {sparse_embeddings.shape} -> ", end='')
                                sparse_embeddings = sparse_embeddings.unsqueeze(0).unsqueeze(0)
                                print(f"{sparse_embeddings.shape}")
                            
                            # バッチサイズ、トークン数、埋め込み次元の確認
                            if sparse_embeddings.dim() == 3:
                                b, t, d = sparse_embeddings.shape
                                print(f"検証時のsparse_embeddings: バッチサイズ={b}, トークン数={t}, 次元={d}")
                            else:
                                print(f"警告: 検証時に3次元でないsparse_embeddings。強制的に3次元に変換します: {sparse_embeddings.shape}")
                                # 次元がまだ3でない場合は強制的に3次元に変換
                                if sparse_embeddings.dim() == 1:
                                    sparse_embeddings = sparse_embeddings.unsqueeze(0).unsqueeze(0)
                                elif sparse_embeddings.dim() == 2:
                                    sparse_embeddings = sparse_embeddings.unsqueeze(1)
                                print(f"変換後: {sparse_embeddings.shape}")
                        except Exception as e:
                            print(f"検証時のsparse_embeddings処理中にエラー: {e}")
                            traceback.print_exc()
                            # 回避策: ダミー埋め込みを作成
                            sparse_embeddings = torch.zeros((1, 1, 256), device=device)
                        
                        # image_embeddingsが辞書型の場合、適切なキーから特徴量を取得
                        if isinstance(image_embeddings, dict):
                            if 'vision_features' in image_embeddings:
                                image_embeddings_tensor = image_embeddings['vision_features']
                            elif 'encoder_embedding' in image_embeddings:
                                image_embeddings_tensor = image_embeddings['encoder_embedding']
                            else:
                                # キーが見つからない場合はキーの一覧を表示
                                print(f"検証時の利用可能なキー: {list(image_embeddings.keys())}")
                                raise ValueError("検証時に適切な画像特徴量キーが見つかりません")
                        else:
                            # すでにテンソルの場合はそのまま使用
                            image_embeddings_tensor = image_embeddings
                        
                        # バッチサイズの一致を確認し、必要に応じて調整
                        img_batch_size = image_embeddings_tensor.shape[0]
                        prompt_batch_size = sparse_embeddings.shape[0]
                        
                        if img_batch_size != prompt_batch_size:
                            print(f"検証時バッチサイズ不一致を調整します: 画像={img_batch_size}, プロンプト={prompt_batch_size}")
                            if img_batch_size > prompt_batch_size:
                                # sparse_embeddingsを複製して同じバッチサイズにする
                                sparse_embeddings = sparse_embeddings.repeat(img_batch_size, 1, 1)
                                # dense_embeddingsはすでに画像バッチサイズと同じ大きさで作成されているので調整不要
                            else:
                                # 先頭のバッチだけを使用
                                image_embeddings_tensor = image_embeddings_tensor[:prompt_batch_size]
                                # dense_embeddingsも同様に調整
                                dense_embeddings = dense_embeddings[:prompt_batch_size]
                                image_pe = image_pe[:prompt_batch_size] if image_pe.dim() > 3 else image_pe
                        
                        # 正しい順序で引数を渡してマスクデコーダを呼び出し
                        mask_predictions = model.sam_mask_decoder(
                            image_embeddings_tensor,  # 画像の埋め込み（テンソル形式に変換済み）
                            image_pe,               # 画像の位置エンコーディング
                            sparse_embeddings,      # スパースプロンプトの埋め込み（3次元に調整済み）
                            dense_embeddings,       # 密なプロンプトの埋め込み
                            False,                  # multimask_output: 単一マスク出力
                            False,                  # repeat_image: 画像の繰り返しなし
                            None                    # high_res_features: デフォルト値を使用
                        )
                    else:
                        mask_predictions = torch.zeros((batch_size, 1, height, width), device=device)
                    
                    # タプルの場合は最初の要素を使用
                    if isinstance(mask_predictions, tuple):
                        mask_predictions = mask_predictions[0]
                    
                    # GTマスクの形状調整
                    if len(gt_mask.shape) == 3:
                        gt_mask = gt_mask.unsqueeze(1)
                    elif len(gt_mask.shape) == 2:
                        gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
                    
                    # マスクサイズの調整
                    if mask_predictions.shape != gt_mask.shape:
                        gt_mask = torch.nn.functional.interpolate(
                            gt_mask.float(), 
                            size=mask_predictions.shape[2:], 
                            mode='nearest'
                        )
                    
                    # 損失計算（検証バリデーション用）- トレーニングに使用した同じ損失関数を使用
                    loss = criterion(mask_predictions, gt_mask.float())
                    loss_batch += loss
                    
                    # バイナリマスクに変換して評価（検証のみ）
                    with torch.no_grad():  # 評価時は勾配計算を無効化
                        pred_mask = (torch.sigmoid(mask_predictions) > 0.5).float()
                        
                        # IoUの計算
                        intersection = torch.sum(pred_mask * gt_mask, dim=(1, 2, 3))
                        union = torch.sum(pred_mask, dim=(1, 2, 3)) + torch.sum(gt_mask, dim=(1, 2, 3)) - intersection
                        iou = (intersection + 1e-8) / (union + 1e-8)
                        
                        # Diceスコアの計算
                        dice = (2 * intersection + 1e-8) / (torch.sum(pred_mask, dim=(1, 2, 3)) + torch.sum(gt_mask, dim=(1, 2, 3)) + 1e-8)
                        
                        iou_scores.extend(iou.cpu().numpy())
                        dice_scores.extend(dice.cpu().numpy())
    
    # 平均スコアの計算
    mean_iou = np.mean(iou_scores)
    mean_dice = np.mean(dice_scores)
    
    # 評価結果をログに記録
    logger.info(f'評価結果サマリー:')
    logger.info(f'- Mean IoU: {mean_iou:.4f}')
    logger.info(f'- Mean Dice: {mean_dice:.4f}')
    logger.info(f'- 総評価サンプル数: {len(iou_scores)}')
    
    # 詳細なメトリクスを記録（サンプル数が多い場合は最初の20個のみ表示）
    display_count = min(20, len(iou_scores))
    if display_count > 0:
        logger.info(f'サンプルごとの詳細メトリクス（最初の{display_count}個）:')
        for i in range(display_count):
            logger.info(f'  サンプル {i}: IoU={iou_scores[i]:.4f}, Dice={dice_scores[i]:.4f}')
    
    print(f'評価結果 - Mean IoU: {mean_iou:.4f}, Mean Dice: {mean_dice:.4f}')
    
    return mean_iou, mean_dice


# メイン実行関数
def main():
    # MPSデバイスの互換性問題を解決するための環境変数設定
    if not os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'):
        print("MPSデバイスでのbicubic補間サポートのため環境変数を設定します")
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # パラメータ設定
    json_path = './manga_data/annotations/manga-frame.json'  # JSONアノテーションファイル
    img_dir = './manga_data/images/'    # 画像ディレクトリ
    # デバイスの設定（MPS: M1/M2 Mac, CUDA: NVIDIA GPU, CPU: その他）
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # トレーニングパラメータ
    batch_size = 2  # 小さいバッチサイズから始める
    num_epochs = 15
    img_size = 1024  # SAM2の推奨入力サイズ
    
    # データ拡張設定
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5)
    ])
    
    # データセットの作成
    full_dataset = MangaSegmentationDataset(json_path, img_dir, transform=transform, img_size=img_size)
    
    # データセットのサイズを表示
    print(f"データセットサイズ: {len(full_dataset)}枚の画像")
    
    # 訓練データとバリデーションデータに分割
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"トレーニングセットサイズ: {train_size}枚")
    print(f"検証セットサイズ: {val_size}枚")
    
    # データローダーの作成（カスタムcollate関数を使用）
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)
    
    # SAM2モデルのロード
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, "checkpoints/sam2.1_hiera_small.pt")
    
    print(f"SAM2モデルを読み込み中: {checkpoint_path}")
    
    # SAM2モデルの構築
    sam2_model = build_sam2(
        config_file="configs/sam2.1/sam2.1_hiera_s.yaml", 
        ckpt_path=checkpoint_path,
        device=device
    )
    
    print("SAM2モデルのロード完了")
    
    # モデルの構造を概観
    print("\n===== SAM2モデルの構造概要 =====")
    for name, module in sam2_model.named_children():
        print(f"トップレベルモジュール: {name} ({type(module).__name__})")
    
    # 推論モードからトレーニングモードに変更
    sam2_model.train()
    
    # LoRAを適用して学習可能なパラメータを設定
    lora_sam2_model = create_lora_sam2(sam2_model)
    
    # ログディレクトリの設定
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 現在の日時を取得してログファイル名を生成
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    training_log_file = os.path.join(log_dir, f"sam2_finetune_{timestamp}.log")
    evaluation_log_file = os.path.join(log_dir, f"sam2_evaluation_{timestamp}.log")
    
    print(f"トレーニングログファイル: {training_log_file}")
    print(f"評価ログファイル: {evaluation_log_file}")
    
    # モデルのトレーニング
    trained_model = train_sam2_for_manga(lora_sam2_model, train_loader, val_loader, device, num_epochs, log_dir)
    
    # モデルの評価
    mean_iou, mean_dice = evaluate_sam2(trained_model, val_loader, device, evaluation_log_file)
    
    # 最終モデルの保存
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'iou_score': mean_iou,
        'dice_score': mean_dice
    }, 'final_sam2_manga_model.pth')
    
    print(f"トレーニング完了! 最終モデルを保存しました。")
    print(f"評価結果 - Mean IoU: {mean_iou:.4f}, Mean Dice: {mean_dice:.4f}")


if __name__ == "__main__":
    main()

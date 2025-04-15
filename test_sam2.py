#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_sam2.py - SAM2モデルを任意の画像で試すスクリプト
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 設定パラメータ（ここを変更して実行時の動作を制御します）
# 処理する画像のパス
IMAGE_PATH = "./test_img/003.jpg"
# 使用するモデルサイズ（"tiny", "small", "base_plus", "large"）
MODEL_SIZE = "large"
# 使用するデバイス（"auto", "mps", "cpu"）
DEVICE = "cpu"  # CPUに変更
# セグメンテーションのためのテキストプロンプト（リスト形式で複数指定可）
PROMPTS = ["an object", "人物"]
# 結果画像の保存先パス（Noneの場合は表示のみ）
OUTPUT_PATH = None

def get_device(device_arg):
    """指定されたデバイス設定に基づいて適切なデバイスを返す"""
    if device_arg == "auto":
        # Apple SiliconのGPUが利用可能か確認
        if torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_arg

def get_model_config(model_size):
    """モデルサイズに基づいて設定ファイルとチェックポイントのパスを返す"""
    model_sizes = {
        "tiny": {
            "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "checkpoint": "./checkpoints/sam2.1_hiera_tiny.pt"
        },
        "small": {
            "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "checkpoint": "./checkpoints/sam2.1_hiera_small.pt"
        },
        "base_plus": {
            "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "checkpoint": "./checkpoints/sam2.1_hiera_base_plus.pt"
        },
        "large": {
            "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
            "checkpoint": "./checkpoints/sam2.1_hiera_large.pt"
        }
    }
    return model_sizes[model_size]

def visualize_results(image_path, masks, prompts, output_path=None):
    """予測結果を可視化"""
    # 元画像の読み込み
    image = np.array(Image.open(image_path).convert("RGB"))
    
    # マスクの可視化
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    if output_path:
        plt.savefig(f"{os.path.splitext(output_path)[0]}_original.jpg")
    else:
        plt.show()
    
    # 各マスクを個別に可視化
    for i, (mask, prompt) in enumerate(zip(masks, prompts)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        
        # マスクのオーバーレイ表示
        mask_array = mask.cpu().numpy()
        colored_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 4), dtype=np.float32)
        colored_mask[mask_array > 0] = [1.0, 0.0, 0.0, 0.5]  # 赤色、半透明
        
        plt.imshow(colored_mask)
        plt.title(f"Mask for prompt: {prompt}")
        plt.axis('off')
        
        if output_path:
            plt.savefig(f"{os.path.splitext(output_path)[0]}_mask_{i}.jpg")
        else:
            plt.show()

def main():
    # 指定された画像が存在するか確認
    if not os.path.exists(IMAGE_PATH):
        print(f"エラー: 指定された画像 '{IMAGE_PATH}' が見つかりません。")
        sys.exit(1)
    
    # モデル設定の取得
    model_config = get_model_config(MODEL_SIZE)
    
    # デバイスの決定
    device = get_device(DEVICE)
    print(f"使用するデバイス: {device}")
    print(f"使用するモデル: {MODEL_SIZE}")
    print(f"画像: {IMAGE_PATH}")
    print(f"プロンプト: {PROMPTS}")
    
    # モデルの読み込み - deviceパラメータを明示的に渡す
    model = build_sam2(model_config["config"], model_config["checkpoint"], device=device)
    predictor = SAM2ImagePredictor(model)
    
    # 推論の実行
    if device == "mps":
        with torch.inference_mode(), torch.autocast("mps", dtype=torch.bfloat16):
            predictor.set_image(IMAGE_PATH)
            masks, _, _ = predictor.predict(PROMPTS)
    else:  # CPU
        with torch.inference_mode():
            predictor.set_image(IMAGE_PATH)
            masks, _, _ = predictor.predict(PROMPTS)
    
    # 結果の可視化
    visualize_results(IMAGE_PATH, masks, PROMPTS, OUTPUT_PATH)
    
    print("処理が完了しました。")

if __name__ == "__main__":
    main()

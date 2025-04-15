#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_sam2_fix.py - SAM2モデルを任意の画像で試すスクリプト（エラー修正版）
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from matplotlib.widgets import RectangleSelector
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 設定パラメータ（ここを変更して実行時の動作を制御します）
# 処理する画像のパス
IMAGE_PATH = "./test_img/004.jpg"
# 使用するモデルサイズ（"tiny", "small", "base_plus", "large"）
MODEL_SIZE = "base_plus"
# 使用するデバイス（"cpu"のみ指定可能に変更）
DEVICE = "cpu"  # Macでのエラー回避のためCPUを使用
# 結果画像の保存先パス（Noneの場合は表示のみ）
OUTPUT_PATH = "./test_result/"

# ユーザーが選択した矩形領域を保存するグローバル変数
selected_box = "test_result"

def select_box(event_click, event_release):
    """ユーザーが選択した矩形領域をグローバル変数に保存する関数"""
    global selected_box
    x1, y1 = int(event_click.xdata), int(event_click.ydata)
    x2, y2 = int(event_release.xdata), int(event_release.ydata)
    selected_box = np.array([x1, y1, x2, y2])
    print(f"選択された矩形領域: {selected_box}")

def get_user_box_selection(image_path):
    """画像を表示し、ユーザーに矩形領域を選択させる関数"""
    global selected_box
    selected_box = None
    
    # 画像を読み込む
    image = np.array(Image.open(image_path).convert("RGB"))
    
    # 画像を表示
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.set_title("画像上でドラッグして矩形領域を選択してください")
    plt.axis('off')
    
    # 矩形選択ツールを設定
    rs = RectangleSelector(
        ax, select_box,
        useblit=True,
        button=[1],  # 左クリックのみ
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True
    )
    
    # ユーザーに指示を表示
    plt.figtext(0.5, 0.01, "矩形領域を選択したら、キーボードの[Enter]キーを押してください", 
                ha="center", fontsize=12, bbox={"boxstyle": "round", "facecolor": "lightgray"})
    
    # キーボードのEnterキーを押した時の処理
    def on_key(event):
        if event.key == 'enter':
            if selected_box is not None:
                plt.close()
            else:
                print("矩形領域が選択されていません。再度選択してください。")
    
    # キーボードイベントを登録
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # 画像を表示して矩形選択を待つ
    plt.tight_layout()
    plt.show()
    
    return selected_box

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

def visualize_results(image_path, masks, output_path=None):
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
    for i, mask in enumerate(masks):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        
        # マスクのオーバーレイ表示
        # 既にnumpy配列なので変換する必要はない
        mask_array = mask if isinstance(mask, np.ndarray) else mask.cpu().numpy()
        colored_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 4), dtype=np.float32)
        colored_mask[mask_array > 0] = [1.0, 0.0, 0.0, 0.5]  # 赤色、半透明
        
        plt.imshow(colored_mask)
        plt.title(f"Mask {i+1}")
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
    
    # ユーザーに矩形領域を選択させる
    print("画像を表示します。矩形領域を選択してください...")
    box = get_user_box_selection(IMAGE_PATH)
    
    if box is None:
        print("矩形領域が選択されませんでした。プログラムを終了します。")
        sys.exit(1)
        
    # モデル設定の取得
    model_config = get_model_config(MODEL_SIZE)
    
    print(f"使用するデバイス: {DEVICE}")
    print(f"使用するモデル: {MODEL_SIZE}")
    print(f"画像: {IMAGE_PATH}")
    print(f"選択された矩形領域: {box}")
    
    # モデルの読み込み - deviceパラメータを明示的に渡す
    model = build_sam2(model_config["config"], model_config["checkpoint"], device=DEVICE)
    predictor = SAM2ImagePredictor(model)
    
    # 画像を読み込む
    input_image = Image.open(IMAGE_PATH).convert("RGB")
    
    # 推論の実行（CPUモードのみ）
    with torch.inference_mode():
        predictor.set_image(input_image)
        # 選択された矩形領域を使用してセグメンテーションを実行
        masks, _, _ = predictor.predict(
            box=box,  # 選択された矩形領域を使用
            multimask_output=True  # 複数のマスク候補を生成
        )
    
    # 結果の可視化
    visualize_results(IMAGE_PATH, masks, OUTPUT_PATH)
    
    print("処理が完了しました。")

if __name__ == "__main__":
    main()

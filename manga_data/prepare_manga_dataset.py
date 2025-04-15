#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
漫画画像のセグメンテーションデータセットを準備するスクリプト
このスクリプトは、既存の漫画画像とアノテーションデータを使用して、
SAM2用のセグメンテーションデータセットを準備します。
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
import random
import shutil
import argparse
import logging
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_panels_from_annotation(image_path, annotation_path, output_dir, visualize=False):
    """
    JSONアノテーションファイルからコマの情報を抽出する関数
    Args:
        image_path: 漫画画像のパス
        annotation_path: アノテーションJSONファイルのパス
        output_dir: 出力ディレクトリ
        visualize: 可視化するかどうか
    """
    # 画像の読み込み
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to read image: {image_path}")
        return False
    
    # アノテーションの読み込み
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load annotation file: {annotation_path}, Error: {e}")
        return False
    
    # アノテーションの形式によって処理を変える必要がある場合があります
    # ここでは、JSONの構造に応じて適切に処理するコードを追加してください
    
    # 結果の可視化
    if visualize:
        img_copy = image.copy()
        
        # ここでアノテーションデータをもとに可視化
        # JSONデータの構造によって処理方法が変わります
        
        vis_path = os.path.join(output_dir, f"{Path(image_path).stem}_annotation.jpg")
        cv2.imwrite(vis_path, img_copy)
    
    # 各コマを個別に保存
    panels_saved = 0
    
    # アノテーションデータから各コマの情報を取得して保存
    # JSONデータの構造によって処理方法が変わります
    
    logger.info(f"Saved {panels_saved} panels from {image_path}")
    return True


def extract_balloons_from_annotation(image_path, annotation_path, output_dir, visualize=False):
    """
    JSONアノテーションファイルから吹き出しの情報を抽出する関数
    Args:
        image_path: 漫画画像のパス
        annotation_path: 吹き出しアノテーションJSONファイルのパス
        output_dir: 出力ディレクトリ
        visualize: 可視化するかどうか
    """
    # 画像の読み込み
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to read image: {image_path}")
        return False
    
    # アノテーションの読み込み
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load balloon annotation file: {annotation_path}, Error: {e}")
        return False
    
    # アノテーションの形式によって処理を変える必要がある場合があります
    # ここでは、JSONの構造に応じて適切に処理するコードを追加してください
    
    # 結果の可視化
    if visualize:
        img_copy = image.copy()
        
        # ここでアノテーションデータをもとに吹き出しを可視化
        # JSONデータの構造によって処理方法が変わります
        
        vis_path = os.path.join(output_dir, f"{Path(image_path).stem}_balloon_annotation.jpg")
        cv2.imwrite(vis_path, img_copy)
    
    # 各吹き出しを個別に保存
    balloons_saved = 0
    
    # アノテーションデータから各吹き出しの情報を取得して保存
    # JSONデータの構造によって処理方法が変わります
    
    logger.info(f"Saved {balloons_saved} balloons from {image_path}")
    return True


def extract_panels_from_manga(image_path, output_dir, min_area=1000, visualize=False):
    """
    漫画画像からコマを抽出する関数（画像処理による自動検出）
    Args:
        image_path: 漫画画像のパス
        output_dir: 出力ディレクトリ
        min_area: 最小コマ面積
        visualize: 可視化するかどうか
    """
    # 画像の読み込み
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to read image: {image_path}")
        return False
    
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ノイズ除去
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2値化
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # モルフォロジー処理（ノイズ除去、コマの境界線を強調）
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 一定サイズ以上の輪郭を抽出
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # 輪郭をソート（左上から右下）
    def contour_sort_key(cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        return y * 1000 + x  # 行優先でソート
    
    valid_contours.sort(key=contour_sort_key)
    
    # 結果の可視化
    if visualize:
        img_copy = image.copy()
        cv2.drawContours(img_copy, valid_contours, -1, (0, 255, 0), 2)
        
        for i, cnt in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(img_copy, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        vis_path = os.path.join(output_dir, f"{Path(image_path).stem}_contours.jpg")
        cv2.imwrite(vis_path, img_copy)
    
    # 各コマを個別に保存
    panels_saved = 0
    
    for i, cnt in enumerate(valid_contours):
        # バウンディングボックスの取得
        x, y, w, h = cv2.boundingRect(cnt)
        
        # コマの切り出し
        panel = image[y:y+h, x:x+w]
        
        # マスクの作成（コマの内部が1、外部が0）
        mask = np.zeros(binary.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        panel_mask = mask[y:y+h, x:x+w]
        
        # コマとマスクの保存
        panel_path = os.path.join(output_dir, "images", f"{Path(image_path).stem}_panel_{i}.jpg")
        mask_path = os.path.join(output_dir, "annotations", f"{Path(image_path).stem}_panel_{i}.png")
        
        cv2.imwrite(panel_path, panel)
        cv2.imwrite(mask_path, panel_mask)
        panels_saved += 1
    
    logger.info(f"Saved {panels_saved} panels from {image_path}")
    return True


def prepare_character_segmentation_data(manga_dir, character_mask_dir, output_dir, split=0.8):
    """
    キャラクター領域のセグメンテーションデータセットを準備する関数
    Args:
        manga_dir: 漫画画像ディレクトリ
        character_mask_dir: キャラクターマスクディレクトリ
        output_dir: 出力ディレクトリ
        split: 訓練/テスト分割比率
    """
    # 出力ディレクトリの作成
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
    
    # 漫画画像とマスクのリストを取得
    manga_files = sorted(list(Path(manga_dir).glob("*.jpg")) + list(Path(manga_dir).glob("*.png")))
    mask_files = sorted(list(Path(character_mask_dir).glob("*.jpg")) + list(Path(character_mask_dir).glob("*.png")))
    
    # ファイル名でマッチング
    paired_files = []
    for manga_file in manga_files:
        manga_name = manga_file.stem
        matching_masks = [m for m in mask_files if manga_name in m.stem]
        
        if matching_masks:
            paired_files.append((manga_file, matching_masks[0]))
    
    # データの分割
    random.seed(42)
    random.shuffle(paired_files)
    
    split_idx = int(len(paired_files) * split)
    train_pairs = paired_files[:split_idx]
    test_pairs = paired_files[split_idx:]
    
    # データの保存
    for dataset, pairs in [("train", train_pairs), ("test", test_pairs)]:
        dataset_dir = os.path.join(output_dir, dataset)
        os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "annotations"), exist_ok=True)
        
        for i, (manga_file, mask_file) in enumerate(tqdm(pairs, desc=f"Processing {dataset} data")):
            # 画像のコピー
            shutil.copy(manga_file, os.path.join(dataset_dir, "images", f"{i:04d}.jpg"))
            
            # マスクの処理
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Failed to read mask: {mask_file}")
                continue
            
            # マスクの二値化（必要に応じて）
            _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # マスクの保存
            cv2.imwrite(os.path.join(dataset_dir, "annotations", f"{i:04d}.png"), mask_binary)
    
    logger.info(f"Dataset prepared: {len(train_pairs)} training samples, {len(test_pairs)} test samples")


def prepare_test_dataset(test_img_dir, output_dir, num_samples=20):
    """
    テスト用データセットを準備する関数（簡易版）
    Args:
        test_img_dir: テスト画像ディレクトリ
        output_dir: 出力ディレクトリ
        num_samples: サンプル数
    """
    # 出力ディレクトリの作成
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
    
    # テスト画像のリストを取得
    test_files = list(Path(test_img_dir).glob("*.jpg")) + list(Path(test_img_dir).glob("*.png"))
    
    # サンプル数の調整
    if len(test_files) > num_samples:
        random.seed(42)
        test_files = random.sample(test_files, num_samples)
    
    # 画像のコピーと疑似マスクの生成
    for i, test_file in enumerate(tqdm(test_files, desc="Preparing test dataset")):
        # 画像のコピー
        shutil.copy(test_file, os.path.join(output_dir, "images", f"test_{i:04d}.jpg"))
        
        # 画像の読み込み
        image = cv2.imread(str(test_file))
        h, w = image.shape[:2]
        
        # 空のマスク（疑似アノテーション）
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.imwrite(os.path.join(output_dir, "annotations", f"test_{i:04d}.png"), empty_mask)
    
    logger.info(f"Test dataset prepared with {len(test_files)} samples")


def generate_synthetic_masks(image_dir, output_dir, num_shapes=5, visualize=True):
    """
    合成マスクを生成する関数
    （実際の漫画データがない場合の代替手段）
    Args:
        image_dir: 画像ディレクトリ
        output_dir: 出力ディレクトリ
        num_shapes: 1画像あたりの図形数
        visualize: 可視化するかどうか
    """
    # 出力ディレクトリの作成
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
    
    if visualize:
        os.makedirs(os.path.join(output_dir, "visualization"), exist_ok=True)
    
    # 画像のリストを取得
    image_files = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    
    # 各画像に対して処理
    for i, image_file in enumerate(tqdm(image_files, desc="Generating synthetic masks")):
        # 画像の読み込み
        image = cv2.imread(str(image_file))
        if image is None:
            logger.warning(f"Failed to read image: {image_file}")
            continue
        
        h, w = image.shape[:2]
        
        # マスクの初期化
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # ランダムな図形を描画
        for _ in range(num_shapes):
            # 図形の種類をランダムに選択 (1: 円, 2: 楕円, 3: 長方形)
            shape_type = random.randint(1, 3)
            
            # ランダムな位置と大きさ
            center_x = random.randint(w // 4, 3 * w // 4)
            center_y = random.randint(h // 4, 3 * h // 4)
            
            if shape_type == 1:  # 円
                radius = random.randint(min(h, w) // 10, min(h, w) // 4)
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)
                
            elif shape_type == 2:  # 楕円
                axis_x = random.randint(w // 10, w // 3)
                axis_y = random.randint(h // 10, h // 3)
                angle = random.randint(0, 180)
                cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), angle, 0, 360, 255, -1)
                
            else:  # 長方形
                width = random.randint(w // 10, w // 3)
                height = random.randint(h // 10, h // 3)
                top_left = (center_x - width // 2, center_y - height // 2)
                bottom_right = (center_x + width // 2, center_y + height // 2)
                cv2.rectangle(mask, top_left, bottom_right, 255, -1)
        
        # 画像とマスクの保存
        img_output_path = os.path.join(output_dir, "images", f"synth_{i:04d}.jpg")
        mask_output_path = os.path.join(output_dir, "annotations", f"synth_{i:04d}.png")
        
        cv2.imwrite(img_output_path, image)
        cv2.imwrite(mask_output_path, mask)
        
        # 可視化
        if visualize:
            # マスクのオーバーレイ表示
            overlay = image.copy()
            color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            color_mask[mask > 0] = [0, 0, 255]  # 赤色
            
            # 画像とマスクを混合
            alpha = 0.5
            cv2.addWeighted(color_mask, alpha, overlay, 1 - alpha, 0, overlay)
            
            # 可視化結果の保存
            vis_path = os.path.join(output_dir, "visualization", f"synth_vis_{i:04d}.jpg")
            cv2.imwrite(vis_path, overlay)
    
    logger.info(f"Generated synthetic masks for {len(image_files)} images")


def process_frame_annotation(json_data, image_file_name, image, image_path, output_dir, visualize=False):
    """
    JSONアノテーションからコマデータを処理する関数
    Args:
        json_data: 読み込まれたJSONデータ
        image_file_name: 画像のファイル名
        image: 画像データ
        image_path: 画像のパス
        output_dir: 出力ディレクトリ
        visualize: 可視化するかどうか
    Returns:
        panels_saved: 保存されたコマの数
    """
    panels_saved = 0
    img_height, img_width = image.shape[:2]
    
    # 可視化用
    if visualize:
        img_copy = image.copy()
    
    try:
        # 特定の画像のデータを検索
        image_data = None
        for img in json_data.get("images", []):
            if img.get("file_name") == image_file_name:
                image_data = img
                break
        
        if not image_data:
            logger.warning(f"Image {image_file_name} not found in annotation data")
            return 0
        
        image_id = image_data.get("id")
        
        # 該当する画像のアノテーションデータを取得
        annotations = []
        for anno in json_data.get("annotations", []):
            if anno.get("image_id") == image_id:
                annotations.append(anno)
        
        if not annotations and image_data.get("num_annotations", 0) > 0:
            logger.warning(f"Image has {image_data['num_annotations']} annotations, but they were not found in the JSON structure")
            # この場合、別の構造でアノテーションが保存されている可能性があるため、
            # 必要に応じて追加の処理を実装してください
        
        # 各アノテーションを処理
        for i, annotation in enumerate(annotations):
            if "segmentation" in annotation:
                # セグメンテーションの処理（ポリゴン形式を想定）
                segmentation = annotation["segmentation"]
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                
                if isinstance(segmentation, list):
                    for seg in segmentation:
                        if isinstance(seg, list) and len(seg) >= 6:
                            # ポリゴン形式のセグメンテーション
                            points = np.array(seg).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(mask, [points], 255)
                
                # バウンディングボックスの取得
                if "bbox" in annotation:
                    bbox = annotation["bbox"]
                    if len(bbox) == 4:
                        x, y, w, h = map(int, bbox)
                    else:
                        logger.warning(f"Invalid bbox format: {bbox}")
                        continue
                else:
                    # マスクからバウンディングボックスを計算
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        x, y, w, h = cv2.boundingRect(contours[0])
                    else:
                        continue
                
                # 可視化
                if visualize:
                    cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img_copy, f"Frame {i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    # マスクの輪郭も描画
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img_copy, contours, -1, (255, 0, 0), 2)
                
                # コマの切り出しとマスクの保存
                panel = image[y:y+h, x:x+w]
                panel_mask = mask[y:y+h, x:x+w]
                
                # コマとマスクの保存
                panel_path = os.path.join(output_dir, "images", f"{Path(image_path).stem}_panel_{i}.jpg")
                mask_path = os.path.join(output_dir, "annotations", f"{Path(image_path).stem}_panel_{i}.png")
                
                cv2.imwrite(panel_path, panel)
                cv2.imwrite(mask_path, panel_mask)
                panels_saved += 1
        
        # 可視化結果の保存
        if visualize and panels_saved > 0:
            vis_path = os.path.join(output_dir, f"{Path(image_path).stem}_frame_annotation.jpg")
            cv2.imwrite(vis_path, img_copy)
            
    except Exception as e:
        logger.error(f"Error processing frame annotation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    return panels_saved


def process_balloon_annotation(json_data, image_file_name, image, image_path, output_dir, visualize=False):
    """
    JSONアノテーションから吹き出しデータを処理する関数
    Args:
        json_data: 読み込まれたJSONデータ
        image_file_name: 画像のファイル名
        image: 画像データ
        image_path: 画像のパス
        output_dir: 出力ディレクトリ
        visualize: 可視化するかどうか
    Returns:
        balloons_saved: 保存された吹き出しの数
    """
    balloons_saved = 0
    img_height, img_width = image.shape[:2]
    
    # 可視化用
    if visualize:
        img_copy = image.copy()
    
    try:
        # 特定の画像のデータを検索
        image_data = None
        for img in json_data.get("images", []):
            if img.get("file_name") == image_file_name:
                image_data = img
                break
        
        if not image_data:
            logger.warning(f"Image {image_file_name} not found in annotation data")
            return 0
        
        image_id = image_data.get("id")
        
        # 該当する画像のアノテーションデータを取得
        annotations = []
        for anno in json_data.get("annotations", []):
            if anno.get("image_id") == image_id:
                annotations.append(anno)
        
        if not annotations:
            # segmentationの形式が異なる場合、直接image_dataから処理する可能性も考慮
            if "segmentation" in image_data:
                annotations = [image_data]
            else:
                logger.warning(f"No annotations found for image {image_file_name}")
                return 0
        
        # 各アノテーションを処理
        for i, annotation in enumerate(annotations):
            if "segmentation" in annotation:
                # セグメンテーションの処理（ポリゴン形式を想定）
                segmentation = annotation["segmentation"]
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                
                if isinstance(segmentation, list):
                    for seg in segmentation:
                        if isinstance(seg, list) and len(seg) >= 6:
                            # ポリゴン形式のセグメンテーション
                            points = np.array(seg).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(mask, [points], 255)
                
                # バウンディングボックスの取得
                if "bbox" in annotation:
                    bbox = annotation["bbox"]
                    if len(bbox) == 4:
                        x, y, w, h = map(int, bbox)
                    else:
                        logger.warning(f"Invalid bbox format: {bbox}")
                        continue
                else:
                    # マスクからバウンディングボックスを計算
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        x, y, w, h = cv2.boundingRect(contours[0])
                    else:
                        continue
                
                # 可視化
                if visualize:
                    cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(img_copy, f"Balloon {i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    # マスクの輪郭も描画
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img_copy, contours, -1, (255, 0, 0), 2)
                
                # 吹き出しの切り出しとマスクの保存
                balloon = image[y:y+h, x:x+w]
                balloon_mask = mask[y:y+h, x:x+w]
                
                # 吹き出しとマスクの保存
                balloon_path = os.path.join(output_dir, "images", f"{Path(image_path).stem}_balloon_{i}.jpg")
                mask_path = os.path.join(output_dir, "annotations", f"{Path(image_path).stem}_balloon_{i}.png")
                
                cv2.imwrite(balloon_path, balloon)
                cv2.imwrite(mask_path, balloon_mask)
                balloons_saved += 1
            elif "num_annotations" in annotation and annotation["num_annotations"] > 0:
                # 別の形式のアノテーションデータ（例：num_annotationsが存在する場合）
                # 詳細なアノテーションデータの形式に応じて処理を追加
                logger.info(f"Found {annotation['num_annotations']} annotations in alternative format")
                # このケースに対応するコードを追加
        
        # 可視化結果の保存
        if visualize and balloons_saved > 0:
            vis_path = os.path.join(output_dir, f"{Path(image_path).stem}_balloon_annotation.jpg")
            cv2.imwrite(vis_path, img_copy)
            
    except Exception as e:
        logger.error(f"Error processing balloon annotation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    return balloons_saved


def extract_annotations_from_json(image_path, frame_json_path, balloon_json_path, output_dir, visualize=False):
    """
    JSONアノテーションファイルからコマと吹き出しの情報を抽出する関数
    Args:
        image_path: 漫画画像のパス
        frame_json_path: コマアノテーションJSONファイルのパス
        balloon_json_path: 吹き出しアノテーションJSONファイルのパス
        output_dir: 出力ディレクトリ
        visualize: 可視化するかどうか
    Returns:
        tuple: (コマの数, 吹き出しの数)
    """
    # 画像の読み込み
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to read image: {image_path}")
        return 0, 0
    
    # 画像のファイル名を取得（パスからではなく）
    image_file_name = os.path.basename(image_path)
    
    panels_saved = 0
    balloons_saved = 0
    
    # コマのアノテーション処理
    try:
        with open(frame_json_path, 'r', encoding='utf-8') as f:
            frame_data = json.load(f)
        
        if frame_data:  # 空でない場合のみ処理
            # process_frame_annotation関数も同様に修正する必要があります
            panels_saved = process_frame_annotation(frame_data, image_file_name, image, image_path, output_dir, visualize)
            logger.info(f"Processed {panels_saved} frames from annotation")
    except Exception as e:
        logger.error(f"Failed to process frame annotation: {e}")
    
    # 吹き出しのアノテーション処理
    try:
        with open(balloon_json_path, 'r', encoding='utf-8') as f:
            balloon_data = json.load(f)
        
        if balloon_data:  # 空でない場合のみ処理
            balloons_saved = process_balloon_annotation(balloon_data, image_file_name, image, image_path, output_dir, visualize)
            logger.info(f"Processed {balloons_saved} balloons from annotation")
    except Exception as e:
        logger.error(f"Failed to process balloon annotation: {e}")
    
    return panels_saved, balloons_saved


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Manga Segmentation Dataset Preparation")
    parser.add_argument("--mode", type=str, 
                        choices=["panel", "character", "test", "synthetic", "annotation"], 
                        default="synthetic", help="Dataset preparation mode")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory with manga images")
    parser.add_argument("--mask_dir", type=str, help="Directory with character masks (for character mode)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the dataset")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples for test mode")
    parser.add_argument("--num_shapes", type=int, default=3, help="Number of shapes for synthetic mode")
    parser.add_argument("--frame_json", type=str, help="Path to frame annotation JSON file")
    parser.add_argument("--balloon_json", type=str, help="Path to balloon annotation JSON file")
    
    args = parser.parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # モードに応じた処理
    if args.mode == "panel":
        # 漫画コマ抽出モード
        os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "annotations"), exist_ok=True)
        
        input_files = list(Path(args.input_dir).glob("*.jpg")) + list(Path(args.input_dir).glob("*.png"))
        
        total_panels = 0
        for img_file in tqdm(input_files, desc="Extracting panels"):
            panels = extract_panels_from_manga(img_file, args.output_dir, 
                                              visualize=args.visualize)
            if panels:
                total_panels += panels
        
        logger.info(f"Extracted {total_panels} panels from {len(input_files)} manga images")
    
    elif args.mode == "annotation":
        # JSONアノテーションからコマと吹き出しを抽出するモード
        if not args.frame_json and not args.balloon_json:
            raise ValueError("At least one of --frame_json or --balloon_json is required for annotation mode")
        
        os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "annotations"), exist_ok=True)
        
        input_files = list(Path(args.input_dir).glob("*.jpg")) + list(Path(args.input_dir).glob("*.png"))
        
        # デフォルトのアノテーションファイルパスを設定
        frame_json_path = args.frame_json or os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                        "annotations", "manga-frame.json")
        balloon_json_path = args.balloon_json or os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                           "annotations", "manga_balloon.json")
        
        total_frames = 0
        total_balloons = 0
        
        for img_file in tqdm(input_files, desc="Processing annotations"):
            frames, balloons = extract_annotations_from_json(
                img_file, frame_json_path, balloon_json_path, 
                args.output_dir, visualize=args.visualize
            )
            total_frames += frames
            total_balloons += balloons
        
        logger.info(f"Processed {total_frames} frames and {total_balloons} balloons from {len(input_files)} manga images")
    
    elif args.mode == "character":
        # キャラクターセグメンテーションモード
        if not args.mask_dir:
            raise ValueError("--mask_dir is required for character mode")
        
        prepare_character_segmentation_data(args.input_dir, args.mask_dir, args.output_dir)
    
    elif args.mode == "test":
        # テストデータセット準備モード
        prepare_test_dataset(args.input_dir, args.output_dir, args.num_samples)
    
    elif args.mode == "synthetic":
        # 合成マスク生成モード
        generate_synthetic_masks(args.input_dir, args.output_dir, 
                                args.num_shapes, args.visualize)


if __name__ == "__main__":
    main()

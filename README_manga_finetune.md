# SAM2 漫画画像セグメンテーション ファインチューニング

このプロジェクトは、SAM2（Segment Anything Model 2）を漫画画像のセグメンテーションタスクに特化させるためのファインチューニング環境です。

## ディレクトリ構成

```
manga_data/         - 漫画データセットのディレクトリ
  ├── images/       - 画像ファイル
  ├── annotations/  - セグメンテーションマスク
  └── prepare_manga_dataset.py  - データセット準備スクリプト
  
manga_finetune/     - ファインチューニング用スクリプト
  ├── finetune_sam2_manga.py    - ファインチューニング実行スクリプト
  └── sam2_manga_config.yaml    - 設定ファイル
  
manga_eval/         - 評価用スクリプト
  └── evaluate_sam2_manga.py    - 評価実行スクリプト
```

## 必要なライブラリ

```bash
pip install torch torchvision matplotlib opencv-python pillow pyyaml tqdm scikit-image
```

## 手順

### 1. データセットの準備

既存の漫画画像からセグメンテーションデータセットを作成します。以下のモードがあります：

#### a. 合成マスクの生成（初期テスト用）

```bash
python manga_data/prepare_manga_dataset.py \
  --mode synthetic \
  --input_dir test_img \
  --output_dir manga_data \
  --num_shapes 3 \
  --visualize
```

#### b. 漫画コマの抽出

```bash
python manga_data/prepare_manga_dataset.py \
  --mode panel \
  --input_dir [漫画画像のディレクトリ] \
  --output_dir manga_data \
  --visualize
```

#### c. キャラクターセグメンテーション

```bash
python manga_data/prepare_manga_dataset.py \
  --mode character \
  --input_dir [漫画画像のディレクトリ] \
  --mask_dir [マスクのディレクトリ] \
  --output_dir manga_data
```

#### d. テストデータセットの準備

```bash
python manga_data/prepare_manga_dataset.py \
  --mode test \
  --input_dir test_img \
  --output_dir manga_data/test \
  --num_samples 20
```

### 2. ファインチューニングの実行

設定ファイル（`manga_finetune/sam2_manga_config.yaml`）を必要に応じて編集してから実行します：

```bash
python manga_finetune/finetune_sam2_manga.py --config manga_finetune/sam2_manga_config.yaml
```

ファインチューニングの主な設定パラメータ：
- `model.backbone`: モデルのバックボーン（"tiny", "small", "base_plus", "large"）
- `training.batch_size`: バッチサイズ
- `training.num_epochs`: エポック数
- `training.learning_rate`: 学習率
- `data.image_size`: 入力画像サイズ

### 3. モデルの評価

ファインチューニング済みモデルを評価します：

```bash
python manga_eval/evaluate_sam2_manga.py \
  --config manga_finetune/sam2_manga_config.yaml \
  --checkpoint [保存されたモデルのパス] \
  --output_dir manga_eval/results
```

## 高度な使用方法

### バックボーンサイズの選択

SAM2には異なるサイズのモデルがあります：
- `tiny`: 最小サイズ、高速だが精度は低め
- `small`: バランスの取れたサイズ
- `base_plus`: 精度とリソースのバランスが良い
- `large`: 最高精度だが計算リソースが必要

モデルサイズはファインチューニング設定ファイルで指定できます：

```yaml
model:
  backbone: "base_plus"  # tiny, small, base_plus, large
```

### データ拡張

データ拡張の設定は以下のように変更できます：

```yaml
data:
  augmentation:
    random_flip: true
    random_rotation: true
    color_jitter: true
    random_resize: [0.8, 1.2]
```

### 損失関数の重み調整

損失関数の重みはセグメンテーション品質に大きく影響します：

```yaml
loss:
  dice_weight: 1.0  # セグメンテーション形状の品質に影響
  focal_weight: 20.0  # クラス不均衡に対応
```

## トラブルシューティング

### CUDA Out of Memory エラー

GPUメモリが不足する場合は、以下の対応を検討してください：
- `batch_size` を小さくする
- 小さいモデル（`tiny` または `small`）を使用する
- 入力画像サイズ（`image_size`）を小さくする
- 勾配蓄積ステップ（`gradient_accumulation_steps`）を増やす

### 過学習の問題

ファインチューニングで過学習が発生する場合は、以下を試してください：
- データ拡張を増やす
- `weight_decay` パラメータを増やす
- 早期終了を実装する

## 注意点

- SAM2は大規模なモデルであり、特に大きいバックボーンでは十分なGPUメモリが必要です。
- ファインチューニングには適切なアノテーションデータが必要です。
- 漫画特有の特徴（線画、白黒、スタイライズされたデザインなど）にモデルを適応させるには、十分なデータ量が必要です。

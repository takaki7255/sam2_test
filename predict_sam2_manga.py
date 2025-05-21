"""
SAM2の学習済みモデルを使用してマンガ画像のセグメンテーション予測を行うスクリプト

このスクリプトは、finetune_sam2_manga.pyで学習したSAM2モデルを使用して、
新しいマンガ画像に対してセグメンテーション予測を行います。
"""

import os
import cv2
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from sam2.build_sam import build_sam2
from peft import get_peft_model, LoraConfig, TaskType


class ResizeLongestSide:
    """画像の長辺をモデルの入力サイズに合わせるリサイザー"""
    
    def __init__(self, target_length):
        self.target_length = target_length
    
    def resize_image(self, image):
        """PIL画像をリサイズして、テンソルに変換します"""
        h, w = image.shape[:2]
        scale = self.target_length / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # リサイズ
        resized_image = cv2.resize(image, (new_w, new_h))
        
        # モデル入力用に正方形にする
        padded_image = np.zeros((self.target_length, self.target_length, 3), dtype=np.uint8)
        padded_image[:new_h, :new_w] = resized_image
        
        return padded_image, (h, w), scale


def load_model(checkpoint_path, device):
    """学習済みモデルを読み込む"""
    print(f"モデルを読み込み中: {checkpoint_path}")
    
    # SAM2の基本モデルを読み込み
    # ファインチューニング時と同じ設定で基本モデルを構築する必要があります
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_ckpt_path = os.path.join(current_dir, "checkpoints/sam2.1_hiera_small.pt")
    config_file = os.path.join(current_dir, "configs/sam2.1/sam2.1_hiera_s.yaml")
    
    base_model = build_sam2(
        config_file=config_file, 
        ckpt_path=base_ckpt_path,
        device=device
    )
    
    # もしLoRAを使用していた場合、LoRA設定を適用
    lora_config = LoraConfig(
        r=16,                     # LoRAのランク
        lora_alpha=32,            # スケーリング係数
        target_modules=["q_proj", "v_proj", "out_proj", "fc1", "fc2"],  # ターゲットモジュール
        lora_dropout=0.1,         # LoRAドロップアウト率
        bias="none",              # バイアスの設定
        task_type=TaskType.TOKEN_CLS  # タスクタイプ
    )
    
    model = get_peft_model(base_model, lora_config)
    
    # 学習済みの重みを読み込み
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # checkpointに直接モデルの状態がある場合
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # LoRAのアダプターのみが保存されている場合
        model.load_state_dict(checkpoint)
    
    model.eval()  # 評価モードに設定
    return model


def predict_mask_from_image(model, image_path, device, img_size=1024, points=None):
    """
    画像からマスクを予測する
    
    Args:
        model: SAM2モデル
        image_path: 画像パス
        device: デバイス
        img_size: 入力画像サイズ
        points: プロンプトポイント（オプション）[[x, y, label], ...]
               label: 1=前景, 0=背景
    """
    # 画像の読み込み
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 画像のリサイズ
    resizer = ResizeLongestSide(img_size)
    input_image, orig_size, scale = resizer.resize_image(image)
    
    # テンソル変換
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    
    # プロンプトポイントの準備（指定がない場合はダミーポイント）
    if points is None:
        # 画像の中央にダミーポイントを配置
        h, w = input_image.shape[:2]
        points = [[w//2, h//2, 1]]  # 中央に前景ポイント
    
    point_coords = torch.tensor([[[p[0] * scale, p[1] * scale] for p in points]], dtype=torch.float32).to(device)
    point_labels = torch.tensor([[p[2] for p in points]], dtype=torch.int).to(device)
    
    # 画像エンベディングの抽出
    with torch.no_grad():
        image_embeddings = model.image_encoder(input_tensor)
        
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
        
        # プロンプトエンコーディング
        if hasattr(model, "prompt_encoder") and hasattr(model.prompt_encoder, "encode_points"):
            sparse_embeddings = model.prompt_encoder.encode_points(
                point_coords,
                point_labels,
                (img_size, img_size)
            )
        else:
            # ダミーの埋め込み
            sparse_embeddings = torch.zeros(1, 1, 256, device=device)
        
        # 位置エンコーディングの取得
        if hasattr(model, "prompt_encoder") and hasattr(model.prompt_encoder, "get_dense_pe"):
            image_pe = model.prompt_encoder.get_dense_pe()
        else:
            # ダミーの位置エンコーディング
            h, w = image_embeddings_tensor.shape[-2:]
            image_pe = torch.zeros((1, 256, h, w), device=device)
        
        # 3次元のスパース埋め込みを確保
        if sparse_embeddings.dim() == 2:
            sparse_embeddings = sparse_embeddings.unsqueeze(1)
        
        # 密な埋め込み
        dense_embeddings = torch.zeros_like(image_embeddings_tensor)
        
        # マスク予測
        mask_predictions = model.sam_mask_decoder(
            image_embeddings_tensor,  # 画像の埋め込み
            image_pe,                 # 画像の位置エンコーディング
            sparse_embeddings,        # スパースプロンプトの埋め込み
            dense_embeddings,         # 密なプロンプトの埋め込み
            False,                    # multimask_output: 単一マスク出力
            False,                    # repeat_image: 画像の繰り返しなし
            None                      # high_res_features: Noneを明示的に使用
        )
    
    # タプルの場合は最初の要素（マスク）を使用
    if isinstance(mask_predictions, tuple):
        mask_predictions = mask_predictions[0]
    
    # シグモイド関数を適用してバイナリマスクに変換
    pred_mask = torch.sigmoid(mask_predictions) > 0.5
    
    # CPU上のnumpy配列に変換
    pred_mask = pred_mask.cpu().squeeze().numpy().astype(np.uint8)
    
    # 元の画像サイズにリサイズ
    if pred_mask.shape[:2] != orig_size:
        pred_mask_resized = cv2.resize(
            pred_mask * 255, 
            (orig_size[1], orig_size[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        pred_mask_resized = pred_mask_resized > 0
    else:
        pred_mask_resized = pred_mask > 0
    
    return image, pred_mask_resized


def visualize_prediction(image, mask, output_path=None):
    """予測結果を可視化"""
    # マスクをカラーマップに変換
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = [0, 255, 0]  # 緑色
    
    # マスクを重ねる
    alpha = 0.5
    blended = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    
    # 3段表示（元画像、マスク、重ね合わせ）
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("元画像")
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("予測マスク")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("マスク重ね合わせ")
    plt.imshow(blended)
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"結果を保存しました: {output_path}")
    
    plt.show()


def process_interactive_points(image):
    """マウスクリックでポイントを指定するための関数"""
    points = []
    current_label = 1  # デフォルトは前景（1）
    
    # マウスイベントのコールバック関数
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, current_label
        
        # 左クリックで前景ポイント追加
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y, 1])
            # ポイントを赤色で描画
            cv2.circle(display_img, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow("ポイント選択（ESC:終了, r:リセット, スペース:背景/前景切替）", display_img)
            print(f"前景ポイント追加: ({x}, {y})")
        
        # 右クリックで背景ポイント追加
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append([x, y, 0])
            # ポイントを青色で描画
            cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("ポイント選択（ESC:終了, r:リセット, スペース:背景/前景切替）", display_img)
            print(f"背景ポイント追加: ({x}, {y})")
    
    # 表示用画像
    display_img = image.copy()
    if display_img.shape[2] == 3:
        display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
    
    # ウィンドウを作成してマウスコールバックを設定
    cv2.namedWindow("ポイント選択（ESC:終了, r:リセット, スペース:背景/前景切替）")
    cv2.setMouseCallback("ポイント選択（ESC:終了, r:リセット, スペース:背景/前景切替）", mouse_callback)
    
    # 説明テキストを表示
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(display_img, "左クリック: 前景ポイント", (10, 30), font, 0.6, (0, 0, 255), 2)
    cv2.putText(display_img, "右クリック: 背景ポイント", (10, 60), font, 0.6, (255, 0, 0), 2)
    cv2.putText(display_img, "r: リセット", (10, 90), font, 0.6, (255, 255, 255), 2)
    cv2.putText(display_img, "ESC: 終了", (10, 120), font, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("ポイント選択（ESC:終了, r:リセット, スペース:背景/前景切替）", display_img)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # ESCキーで終了
        if key == 27:
            break
        
        # rキーでリセット
        if key == ord('r'):
            points = []
            display_img = image.copy()
            if display_img.shape[2] == 3:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
            
            # 説明テキストを再表示
            cv2.putText(display_img, "左クリック: 前景ポイント", (10, 30), font, 0.6, (0, 0, 255), 2)
            cv2.putText(display_img, "右クリック: 背景ポイント", (10, 60), font, 0.6, (255, 0, 0), 2)
            cv2.putText(display_img, "r: リセット", (10, 90), font, 0.6, (255, 255, 255), 2)
            cv2.putText(display_img, "ESC: 終了", (10, 120), font, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("ポイント選択（ESC:終了, r:リセット, スペース:背景/前景切替）", display_img)
            print("ポイントをリセットしました")
        
        # スペースキーで前景/背景を切り替え
        if key == 32:  # スペースキー
            current_label = 1 - current_label
            label_text = "前景" if current_label == 1 else "背景"
            print(f"現在のモード: {label_text}ポイント")
    
    cv2.destroyAllWindows()
    return points


def main():
    parser = argparse.ArgumentParser(description="SAM2モデルを使用してマンガ画像のセグメンテーション予測")
    parser.add_argument("--image", type=str, help="入力画像のパス")
    parser.add_argument("--model", type=str, default="best_sam2_manga_model.pth", help="学習済みモデルのパス")
    parser.add_argument("--output", type=str, default="prediction_result.png", help="出力画像のパス")
    parser.add_argument("--batch", action="store_true", help="フォルダ内の全画像を処理")
    parser.add_argument("--interactive", action="store_true", help="インタラクティブモード（マウスでポイント指定）")
    parser.add_argument("--img_size", type=int, default=1024, help="入力画像サイズ")
    
    args = parser.parse_args()
    
    # MPSデバイスの互換性問題を解決するための環境変数設定
    if not os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'):
        print("MPSデバイスでのbicubic補間サポートのため環境変数を設定します")
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # デバイスの設定
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # モデルの読み込み
    model = load_model(args.model, device)
    
    if args.batch:
        # バッチモード：フォルダ内の全画像を処理
        if not args.image or not os.path.isdir(args.image):
            print("バッチモードの場合は--imageに有効なディレクトリパスを指定してください")
            return
        
        # 出力ディレクトリの作成
        output_dir = "batch_prediction_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # サポートされる画像拡張子
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # フォルダ内の画像を処理
        image_files = [f for f in os.listdir(args.image) 
                      if os.path.isfile(os.path.join(args.image, f)) and 
                      any(f.lower().endswith(ext) for ext in image_extensions)]
        
        for img_file in tqdm(image_files, desc="画像処理中"):
            image_path = os.path.join(args.image, img_file)
            output_path = os.path.join(output_dir, f"pred_{os.path.splitext(img_file)[0]}.png")
            
            try:
                image, mask = predict_mask_from_image(model, image_path, device, args.img_size)
                
                # マスクの可視化と保存
                colored_mask = np.zeros_like(image)
                colored_mask[mask > 0] = [0, 255, 0]  # 緑色
                blended = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)
                
                # BGR変換して保存
                cv2.imwrite(output_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"画像 {img_file} の処理中にエラーが発生しました: {e}")
        
        print(f"バッチ処理が完了しました。結果は {output_dir} ディレクトリに保存されています。")
    
    else:
        # 単一画像モード
        if not args.image or not os.path.isfile(args.image):
            print("--imageに有効な画像ファイルパスを指定してください")
            return
        
        points = None
        
        # インタラクティブモード
        if args.interactive:
            # 画像を読み込み
            image = cv2.imread(args.image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # ユーザーにポイントを選択させる
            points = process_interactive_points(image)
            print(f"選択されたポイント: {points}")
        
        # マスク予測
        image, mask = predict_mask_from_image(model, args.image, device, args.img_size, points)
        
        # 結果の可視化と保存
        visualize_prediction(image, mask, args.output)


if __name__ == "__main__":
    main()

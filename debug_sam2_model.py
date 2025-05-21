"""
SAM2モデルの構造をデバッグするためのスクリプト
このスクリプトはSAM2モデルの内部構造を詳細に分析し、マスク予測に必要な処理フローを検証します
"""

import os
import torch
import inspect
from sam2.build_sam import build_sam2

def debug_sam2_model():
    # MPSデバイスの互換性問題を解決するための環境変数設定
    if not os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'):
        print("MPSデバイスでのbicubic補間サポートのため環境変数を設定します")
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # デバイスの設定
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # SAM2モデルのロード
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, "checkpoints/sam2.1_hiera_small.pt")
    
    # build_sam2関数を使用
    print("SAM2モデルをロード中...")
    model = build_sam2(
        config_file="configs/sam2.1/sam2.1_hiera_s.yaml", 
        ckpt_path=checkpoint_path,
        device=device
    )
    
    # モデルの構造を詳細に分析
    print("\n===== SAM2モデルの構造詳細分析 =====")
    for name, module in model.named_children():
        print(f"トップレベルモジュール: {name} ({type(module).__name__})")
        for sub_name, sub_module in module.named_children():
            print(f"  サブモジュール: {name}.{sub_name} ({type(sub_module).__name__})")
            for sub_sub_name, sub_sub_module in sub_module.named_children():
                print(f"    サブサブモジュール: {name}.{sub_name}.{sub_sub_name} ({type(sub_sub_module).__name__})")
    
    # モジュールの関数シグネチャを分析
    print("\n===== 重要なモジュールの関数シグネチャ =====")
    modules_to_check = [
        ("image_encoder", model.image_encoder if hasattr(model, "image_encoder") else None),
        ("prompt_encoder", model.prompt_encoder if hasattr(model, "prompt_encoder") else None),
        ("mask_decoder", model.mask_decoder if hasattr(model, "mask_decoder") else None),
        ("sam_mask_decoder", model.sam_mask_decoder if hasattr(model, "sam_mask_decoder") else None),
    ]
    
    for name, module in modules_to_check:
        if module is not None:
            try:
                sig = inspect.signature(module.forward)
                print(f"{name}の引数: {list(sig.parameters.keys())}")
            except Exception as e:
                print(f"{name}のシグネチャ取得エラー: {e}")
    
    # ダミーデータでの動作確認テスト
    print("\n===== ダミーデータでのモデル機能テスト =====")
    # ダミー入力の作成
    dummy_image = torch.rand(1, 3, 1024, 1024, device=device)
    dummy_points = torch.tensor([[[500, 500]]], dtype=torch.float, device=device)
    dummy_labels = torch.ones(1, 1, dtype=torch.int, device=device)
    
    # 画像エンコーダのテスト
    try:
        with torch.no_grad():
            print("画像エンコーダをテスト中...")
            dummy_embeddings = model.image_encoder(dummy_image)
            print(f"画像エンコーダテスト成功: {dummy_embeddings.keys() if isinstance(dummy_embeddings, dict) else type(dummy_embeddings)}")
            if isinstance(dummy_embeddings, dict):
                for k, v in dummy_embeddings.items():
                    print(f"  - {k}: 形状={v.shape if isinstance(v, torch.Tensor) else [item.shape for item in v]}")
        
        # プロンプトエンコーダのテスト (もし存在すれば)
        if hasattr(model, "prompt_encoder"):
            print("\nプロンプトエンコーダをテスト中...")
            try:
                # 方法1: encode_points
                if hasattr(model.prompt_encoder, "encode_points"):
                    sparse_embeddings = model.prompt_encoder.encode_points(
                        dummy_points, dummy_labels, (1024, 1024)
                    )
                    print(f"encode_points成功: 形状={sparse_embeddings.shape}")
                
                # 方法2: 直接forward
                try:
                    result = model.prompt_encoder(
                        points=dummy_points,
                        labels=dummy_labels,
                        boxes=None,
                        masks=None,
                    )
                    if isinstance(result, tuple):
                        print(f"prompt_encoder.forward成功: 要素数={len(result)}")
                        for i, r in enumerate(result):
                            print(f"  - 要素{i}: 形状={r.shape if isinstance(r, torch.Tensor) else type(r)}")
                    else:
                        print(f"prompt_encoder.forward成功: 形状={result.shape}")
                except Exception as e:
                    print(f"prompt_encoder.forward失敗: {e}")
                
                # マスクデコーダをテスト
                print("\nマスクデコーダをテスト中...")
                if hasattr(model, "mask_decoder"):
                    try:
                        # テスト1: 基本的な呼び出し
                        vision_features = dummy_embeddings.get('vision_features', None)
                        if vision_features is not None:
                            try:
                                mask_result = model.mask_decoder(
                                    image_embeddings=dummy_embeddings,
                                    prompt_embeddings=sparse_embeddings,
                                )
                                print(f"mask_decoder基本呼び出し成功: {type(mask_result)}")
                                if isinstance(mask_result, tuple):
                                    for i, r in enumerate(mask_result):
                                        print(f"  - 要素{i}: 形状={r.shape if isinstance(r, torch.Tensor) else type(r)}")
                                else:
                                    print(f"  - 形状={mask_result.shape}")
                            except Exception as e:
                                print(f"mask_decoder基本呼び出し失敗: {e}")
                                
                                # テスト2: 代替引数での呼び出し
                                try:
                                    # シグネチャを確認して代替方法を試す
                                    sig_params = list(inspect.signature(model.mask_decoder.forward).parameters.keys())
                                    print(f"mask_decoder引数: {sig_params}")
                                    
                                    # 一般的な代替パターン
                                    kwargs = {}
                                    if 'image_embeddings' in sig_params:
                                        kwargs['image_embeddings'] = dummy_embeddings
                                    if 'prompt_embeddings' in sig_params:
                                        kwargs['prompt_embeddings'] = sparse_embeddings
                                    if 'sparse_embeddings' in sig_params:
                                        kwargs['sparse_embeddings'] = sparse_embeddings
                                    if 'dense_embeddings' in sig_params:
                                        kwargs['dense_embeddings'] = torch.zeros(1, 256, device=device)
                                    
                                    mask_result = model.mask_decoder(**kwargs)
                                    print(f"mask_decoder代替呼び出し成功: {type(mask_result)}")
                                except Exception as e2:
                                    print(f"mask_decoder代替呼び出し失敗: {e2}")
                    except Exception as e:
                        print(f"マスクデコーダテスト全体エラー: {e}")
            except Exception as e:
                print(f"プロンプトエンコーダテスト失敗: {e}")
    except Exception as e:
        print(f"画像エンコーダテスト失敗: {e}")
    
    print("===== テスト完了 =====")

if __name__ == "__main__":
    debug_sam2_model()

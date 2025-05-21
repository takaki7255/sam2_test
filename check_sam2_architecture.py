#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import inspect
from sam2.build_sam import build_sam2

# MPSデバイスのフォールバックを有効化
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# デバイスの設定
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")

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

# モデルの構造を調査
if hasattr(sam2_model, "sam_mask_decoder"):
    print("\n=== sam_mask_decoder の構造を調査 ===")
    signature = inspect.signature(sam2_model.sam_mask_decoder.forward)
    print(f"パラメータ: {list(signature.parameters.keys())}")
    
    for param_name, param in signature.parameters.items():
        print(f"  {param_name}: {param.annotation}")
        if param.default is not inspect.Parameter.empty:
            print(f"    デフォルト値: {param.default}")
elif hasattr(sam2_model, "mask_decoder"):
    print("\n=== mask_decoder の構造を調査 ===")
    signature = inspect.signature(sam2_model.mask_decoder.forward)
    print(f"パラメータ: {list(signature.parameters.keys())}")
    
    for param_name, param in signature.parameters.items():
        print(f"  {param_name}: {param.annotation}")
        if param.default is not inspect.Parameter.empty:
            print(f"    デフォルト値: {param.default}")
else:
    print("モデルにマスクデコーダが見つかりません")

# SAM2のメソッドを確認
print("\n=== SAM2モデルのメソッド ===")
for method_name in dir(sam2_model):
    if not method_name.startswith("_"):
        method = getattr(sam2_model, method_name)
        if callable(method):
            try:
                sig = inspect.signature(method)
                print(f"{method_name}{sig}")
            except (ValueError, TypeError):
                print(f"{method_name}: シグネチャ取得不可")

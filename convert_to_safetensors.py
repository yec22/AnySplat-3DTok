#!/usr/bin/env python3
"""将 PyTorch Lightning checkpoint 转换为 Hugging Face safetensors 格式

用法:
    python convert_to_safetensors.py \
      --ckpt /personal/AnySplat_query_v2/output/exp_multidataset/64k_query/checkpoints/epoch_213-step_22000.ckpt \
      --output-dir /personal/AnySplat_query_v2/output/exp_multidataset_ckpt

生成的文件结构:
    output-dir/
        ├── model.safetensors      # 主模型权重
        └── config.json            # 模型配置信息 (包含 encoder_cfg 和 decoder_cfg)

在 inference.py 中使用:
    model = AnySplat.from_pretrained("/path/to/output-dir")
"""

import json
import argparse
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file


def load_model_config(config_path: Path):
    """从 Hydra config.yaml 加载模型配置"""
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)

    model_config = full_config.get('model', {})

    return {
        "encoder_cfg": model_config.get('encoder', {}),
        "decoder_cfg": model_config.get('decoder', {}),
    }


def convert_ckpt_to_safetensors(ckpt_path: str, output_dir: str, config_path: str = None):
    """将 PyTorch Lightning checkpoint 转换为 safetensor 格式"""

    ckpt_path = Path(ckpt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # 获取 state_dict
    state_dict = ckpt.get('state_dict', ckpt)

    # 移除 "model." 前缀（Lightning Module 包装层）
    # 因为 AnySplat 的 state_dict 键名是 encoder.xxx 和 decoder.xxx
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_key = k[6:]  # 去掉 "model."
        else:
            new_key = k
        new_state_dict[new_key] = v

    # 保存为 safetensors
    model_path = output_dir / "model.safetensors"
    save_file(new_state_dict, model_path)
    print(f"Saved model.safetensors: {model_path}")
    print(f"  - Total parameters: {len(new_state_dict)}")
    print(f"  - Keys (first 5): {list(new_state_dict.keys())[:5]}")

    # 加载或推断配置路径
    if config_path is None:
        # 默认在 checkpoint 所在目录的 .hydra/config.yaml
        possible_config = ckpt_path.parent.parent / ".hydra" / "config.yaml"
        if possible_config.exists():
            config_path = possible_config
            print(f"Auto-detected config: {config_path}")
        else:
            raise ValueError(
                f"Could not find config file. Please specify --config. "
                f"Tried: {possible_config}"
            )
    else:
        config_path = Path(config_path)

    # 加载模型配置
    print(f"Loading model config from: {config_path}")
    model_cfg = load_model_config(config_path)

    # 保存 config.json
    # 关键：需要包含 encoder_cfg 和 decoder_cfg 以便 from_pretrained 正确初始化
    config = {
        "model_type": "anysplat",
        "original_checkpoint": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        **model_cfg,  # encoder_cfg and decoder_cfg
    }

    config_path_out = output_dir / "config.json"
    with open(config_path_out, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config.json: {config_path_out}")
    print("\n✅ 转换完成！")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch Lightning checkpoint to safetensors"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the .ckpt file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to Hydra config.yaml (default: auto-detect from ../.hydra/config.yaml)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: checkpoint_dir/converted)"
    )

    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise ValueError(f"Checkpoint not found: {ckpt_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ckpt_path.parent / "converted"

    convert_ckpt_to_safetensors(str(ckpt_path), str(output_dir), args.config)


if __name__ == "__main__":
    main()
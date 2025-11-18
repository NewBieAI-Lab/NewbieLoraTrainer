#!/usr/bin/env python3
"""Newbie LoRA 权重合并工具 - 将 LoRA 适配器合并回基础模型"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent))
import models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("newbie_lora_merger")


def load_base_model(base_model_path: str, trust_remote_code: bool = True):
    """加载基础模型 (仅支持 diffusers 格式)"""
    logger.info(f"Loading base model: {base_model_path}")

    model_index_path = os.path.join(base_model_path, "model_index.json")
    if not os.path.exists(model_index_path):
        raise ValueError("Only diffusers format supported. Run convert_newbie_to_diffusers.py first.")

    text_encoder_path = os.path.join(base_model_path, "text_encoder")
    text_encoder = AutoModel.from_pretrained(text_encoder_path, torch_dtype=torch.float32, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "right"

    clip_model_path = os.path.join(base_model_path, "clip_model")
    clip_model = AutoModel.from_pretrained(clip_model_path, torch_dtype=torch.float32, trust_remote_code=True)
    clip_tokenizer = AutoTokenizer.from_pretrained(clip_model_path, trust_remote_code=True)

    transformer_path = os.path.join(base_model_path, "transformer")
    with open(os.path.join(transformer_path, "config.json"), 'r') as f:
        model_config = json.load(f)

    text_encoder_config = AutoConfig.from_pretrained(text_encoder_path, trust_remote_code=trust_remote_code)
    cap_feat_dim = text_encoder_config.text_config.hidden_size
    model_name = model_config.get('_class_name', 'NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP')

    model = models.__dict__[model_name](
        in_channels=model_config.get('in_channels', 16),
        qk_norm=True,
        cap_feat_dim=cap_feat_dim,
        clip_text_dim=model_config.get('clip_text_dim', 1024),
        clip_img_dim=model_config.get('clip_img_dim', 1024)
    )

    weight_path = os.path.join(transformer_path, "diffusion_pytorch_model.safetensors")
    if os.path.exists(weight_path):
        state_dict = load_file(weight_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys when loading base model: {len(missing_keys)}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading base model: {len(unexpected_keys)}")

    vae_path = os.path.join(base_model_path, "vae")
    vae = AutoencoderKL.from_pretrained(
        vae_path if os.path.exists(vae_path) else "stabilityai/sdxl-vae",
        torch_dtype=torch.float32
    )

    logger.info("Base model loaded")
    return model, vae, text_encoder, tokenizer, clip_model, clip_tokenizer


def merge_lora_weights(base_model_path: str, lora_path: str, output_path: str,
                       trust_remote_code: bool = True, device: str = "cpu"):
    """合并 LoRA 权重到基础模型并保存为 diffusers 格式"""
    logger.info(f"Merging LoRA: {lora_path} + {base_model_path} -> {output_path}")
    os.makedirs(output_path, exist_ok=True)

    model, vae, text_encoder, tokenizer, clip_model, clip_tokenizer = load_base_model(base_model_path, trust_remote_code)

    logger.info(f"Loading LoRA weights from: {lora_path}")
    try:
        lora_model = PeftModel.from_pretrained(model, lora_path)
        logger.info("LoRA adapter loaded, merging weights...")
        merged_model = lora_model.merge_and_unload()
        logger.info("LoRA weights merged successfully")
    except Exception as e:
        logger.error(f"Failed to merge LoRA: {e}")
        import traceback
        traceback.print_exc()
        raise

    try:
        # 保存 Transformer
        transformer_path = os.path.join(output_path, "transformer")
        os.makedirs(transformer_path, exist_ok=True)
        save_file(merged_model.state_dict(), os.path.join(transformer_path, "diffusion_pytorch_model.safetensors"))

        config = {
            "model_type": "nextdit_clip", "patch_size": 2, "dim": 2304, "n_layers": 36, "n_heads": 24,
            "n_kv_heads": 8, "in_channels": 16, "axes_dims": [32, 32, 32], "axes_lens": [1024, 512, 512],
            "clip_text_dim": 1024, "clip_img_dim": 1024, "_class_name": "NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP"
        }
        with open(os.path.join(transformer_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # 保存其他组件
        text_encoder.save_pretrained(os.path.join(output_path, "text_encoder"), safe_serialization=True)
        tokenizer.save_pretrained(os.path.join(output_path, "text_encoder"))

        clip_model.save_pretrained(os.path.join(output_path, "clip_model"), safe_serialization=True)
        clip_tokenizer.save_pretrained(os.path.join(output_path, "clip_model"))

        vae.save_pretrained(os.path.join(output_path, "vae"), safe_serialization=True)

        # 创建模型索引
        model_index = {
            "_class_name": "NewbiePipeline",
            "_diffusers_version": "0.30.0",
            "transformer": ["transformer", "NextDiT_CLIP"],
            "text_encoder": ["text_encoder", "Gemma3Model"],
            "tokenizer": ["text_encoder", "AutoTokenizer"],
            "clip_model": ["clip_model", "JinaCLIPModel"],
            "clip_tokenizer": ["clip_model", "AutoTokenizer"],
            "vae": ["vae", "AutoencoderKL"]
        }
        with open(os.path.join(output_path, "model_index.json"), "w") as f:
            json.dump(model_index, f, indent=2)

        logger.info(f"Merge complete! Saved to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Newbie LoRA Merge Tool")
    parser.add_argument("--base_model_path", required=True, help="Base model path (diffusers format)")
    parser.add_argument("--lora_path", required=True, help="Trained LoRA adapter path")
    parser.add_argument("--output_path", required=True, help="Output path for merged model")
    parser.add_argument("--trust_remote_code", action="store_true", default=True, help="Trust remote code")
    parser.add_argument("--device", default="cpu", help="Device (cuda/cpu)")

    args = parser.parse_args()

    merge_lora_weights(
        base_model_path=args.base_model_path,
        lora_path=args.lora_path,
        output_path=args.output_path,
        trust_remote_code=args.trust_remote_code,
        device=args.device
    )


if __name__ == "__main__":
    main()

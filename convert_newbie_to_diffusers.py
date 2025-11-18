#!/usr/bin/env python3
"""Newbie 模型转 Diffusers 格式工具"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer
from diffusers import AutoencoderKL

sys.path.insert(0, str(Path(__file__).parent.parent))
import models


def parse_args():
    parser = argparse.ArgumentParser(description="将Newbie (3B CLIP)模型转换为Diffusers格式")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型checkpoint的.pth文件路径 (例如: consolidated.00-of-01.pth)"
    )
    parser.add_argument(
        "--gemma3",
        type=str,
        default="google/gemma-3-4b-it",
        help="Gemma3模型的HuggingFace repo或本地路径 (默认: google/gemma-3-4b-it)"
    )
    parser.add_argument(
        "--jina",
        type=str,
        default="jinaai/jina-clip-v2",
        help="Jina CLIP模型的HuggingFace repo或本地路径 (默认: jinaai/jina-clip-v2)"
    )
    parser.add_argument(
        "--vae",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="VAE模型的HuggingFace repo或本地路径 (默认: black-forest-labs/FLUX.1-dev，使用其中的VAE子文件夹)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="模型精度 (默认: bf16)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录 (默认: 与checkpoint同目录下的diffusers文件夹)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP",
        help="模型名称 (默认: NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP)"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token用于下载受限模型"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="是否将转换后的模型推送到HuggingFace Hub"
    )
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        default=None,
        help="HuggingFace Hub repo ID (例如: username/newbie-diffusers)"
    )

    return parser.parse_args()


def get_dtype(dtype_str):
    """将字符串转换为torch dtype"""
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32
    }
    return dtype_map[dtype_str]


def load_checkpoint(checkpoint_path):
    """加载模型checkpoint"""
    print(f"\n{'='*60}")
    print("📂 加载Checkpoint")
    print(f"{'='*60}")
    print(f"路径: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 提取state_dict
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("✅ 从'model'键加载state_dict")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("✅ 从'state_dict'键加载state_dict")
        else:
            state_dict = checkpoint
            print("✅ 直接使用checkpoint作为state_dict")
    else:
        state_dict = checkpoint
        print("✅ 直接使用checkpoint作为state_dict")

    print(f"📊 参数总数: {len(state_dict)}")

    return state_dict


def load_text_encoder(gemma3_path, dtype, hf_token=None):
    """加载Gemma3文本编码器"""
    print(f"\n{'='*60}")
    print("📝 加载Gemma3文本编码器")
    print(f"{'='*60}")
    print(f"路径: {gemma3_path}")
    print(f"精度: {dtype}")

    text_encoder = AutoModel.from_pretrained(
        gemma3_path,
        torch_dtype=dtype,
        token=hf_token
    )

    tokenizer = AutoTokenizer.from_pretrained(
        gemma3_path,
        token=hf_token
    )
    tokenizer.padding_side = "right"

    cap_feat_dim = text_encoder.config.text_config.hidden_size
    print(f"✅ Gemma3加载成功")
    print(f"📊 隐藏维度: {cap_feat_dim}")

    return text_encoder, tokenizer, cap_feat_dim


def load_clip_model(jina_path, dtype, hf_token=None):
    """加载Jina CLIP模型"""
    print(f"\n{'='*60}")
    print("🖼️  加载Jina CLIP模型")
    print(f"{'='*60}")
    print(f"路径: {jina_path}")
    print(f"精度: {dtype}")

    from transformers import AutoConfig

    clip_config = AutoConfig.from_pretrained(
        jina_path,
        trust_remote_code=True
    )

    clip_tokenizer = AutoTokenizer.from_pretrained(
        jina_path,
        trust_remote_code=True
    )

    clip_model = AutoModel.from_pretrained(
        jina_path,
        torch_dtype=dtype,
        config=clip_config,
        trust_remote_code=True,
        token=hf_token
    )

    print(f"✅ Jina CLIP加载成功")

    return clip_model, clip_tokenizer, clip_config


def load_vae(vae_path, dtype, hf_token=None):
    """加载VAE模型"""
    print(f"\n{'='*60}")
    print("🎨 加载VAE模型")
    print(f"{'='*60}")
    print(f"路径: {vae_path}")
    print(f"精度: {dtype}")

    # 检查是否是包含vae子文件夹的完整模型路径（如FLUX）
    vae_subfolder_path = os.path.join(vae_path, "vae") if os.path.isdir(vae_path) else None

    if vae_subfolder_path and os.path.exists(os.path.join(vae_subfolder_path, "config.json")):
        print(f"检测到VAE子文件夹，从 {vae_subfolder_path} 加载")
        vae = AutoencoderKL.from_pretrained(
            vae_subfolder_path,
            torch_dtype=dtype,
            token=hf_token
        )
    else:
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if not vae_path.endswith("vae") else None,
            torch_dtype=dtype,
            token=hf_token
        )

    print(f"✅ VAE加载成功")
    print(f"📊 潜在空间通道数: {vae.config.latent_channels}")

    return vae


def create_model(model_name, cap_feat_dim, qk_norm=True):
    """创建NextDiT CLIP模型"""
    print(f"\n{'='*60}")
    print("🏗️  创建NextDiT CLIP模型")
    print(f"{'='*60}")
    print(f"模型: {model_name}")
    print(f"cap_feat_dim: {cap_feat_dim}")
    print(f"qk_norm: {qk_norm}")

    if model_name not in models.__dict__:
        raise ValueError(f"模型 '{model_name}' 不存在。可用模型: {list(models.__dict__.keys())}")

    model = models.__dict__[model_name](
        in_channels=16,
        qk_norm=qk_norm,
        cap_feat_dim=cap_feat_dim,
    )

    param_count = model.parameter_count()
    print(f"✅ 模型创建成功")
    print(f"📊 参数量: {param_count:,}")

    return model


def convert_to_diffusers(
    model,
    text_encoder,
    tokenizer,
    clip_model,
    clip_tokenizer,
    vae,
    output_dir,
    dtype
):
    """将模型转换为Diffusers格式"""
    print(f"\n{'='*60}")
    print("🔄 转换为Diffusers格式")
    print(f"{'='*60}")
    print(f"输出目录: {output_dir}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 保存DiT模型
    dit_path = output_path / "transformer"
    dit_path.mkdir(exist_ok=True)

    print("\n📦 保存DiT Transformer...")
    # 保存为safetensors格式
    model_state_dict = model.state_dict()
    save_file(model_state_dict, dit_path / "diffusion_pytorch_model.safetensors")

    # 保存模型配置
    model_config = {
        "model_type": "nextdit_clip",
        "patch_size": 2,
        "dim": 2304,
        "n_layers": 36,
        "n_heads": 24,
        "n_kv_heads": 8,
        "in_channels": 16,
        "axes_dims": [32, 32, 32],
        "axes_lens": [1024, 512, 512],
        "clip_text_dim": 1024,
        "clip_img_dim": 1024,
        "_class_name": "NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP",
    }

    with open(dit_path / "config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    print(f"✅ DiT模型已保存到: {dit_path}")

    # 2. 保存文本编码器 (Gemma3)
    text_encoder_path = output_path / "text_encoder"
    text_encoder_path.mkdir(exist_ok=True)

    print("\n📝 保存Gemma3文本编码器...")
    text_encoder.save_pretrained(text_encoder_path, safe_serialization=True)
    tokenizer.save_pretrained(text_encoder_path)
    print(f"✅ Gemma3已保存到: {text_encoder_path}")

    # 3. 保存CLIP模型
    clip_path = output_path / "clip_model"
    clip_path.mkdir(exist_ok=True)

    print("\n🖼️  保存Jina CLIP模型...")
    clip_model.save_pretrained(clip_path, safe_serialization=True)
    clip_tokenizer.save_pretrained(clip_path)
    print(f"✅ Jina CLIP已保存到: {clip_path}")

    # 4. 保存VAE
    vae_path = output_path / "vae"
    vae_path.mkdir(exist_ok=True)

    print("\n🎨 保存VAE...")
    vae.save_pretrained(vae_path, safe_serialization=True)
    print(f"✅ VAE已保存到: {vae_path}")

    # 5. 创建模型索引和配置
    model_index = {
        "_class_name": "NewbiePipeline",
        "_diffusers_version": "0.30.0",
        "transformer": ["transformer", "NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP"],
        "text_encoder": ["text_encoder", "Gemma3ForConditionalGeneration"],
        "tokenizer": ["text_encoder", "AutoTokenizer"],
        "clip_model": ["clip_model", "JinaCLIPModel"],
        "clip_tokenizer": ["clip_model", "AutoTokenizer"],
        "vae": ["vae", "AutoencoderKL"],
    }

    with open(output_path / "model_index.json", "w") as f:
        json.dump(model_index, f, indent=2)

    print(f"\n✅ 模型索引已创建")

    # 5. 创建README
    readme_content = f"""---
license: apache-2.0
tags:
- text-to-image
- diffusion
- nextdit
- lumina
library_name: diffusers
pipeline_tag: text-to-image
---

# Newbie (3B CLIP) - Diffusers Format

这是Newbie AI的NextDiT 3B CLIP模型的Diffusers格式版本。

## 模型架构

- **Transformer**: NextDiT (3B参数)
  - 36层 Transformer blocks
  - 24个注意力头，8个KV头 (GQA)
  - 隐藏维度: 2304
  - Patch size: 2

- **文本编码器**: Gemma-3-4B-IT
  - 用于提取文本特征和条件生成

- **CLIP模型**: Jina CLIP v2
  - 提供额外的文本-图像对齐特征
  - 增强文本理解能力

## 使用方法

```python
import torch
from diffusers import DiffusionPipeline

# 加载管道
pipe = DiffusionPipeline.from_pretrained(
    "path/to/this/model",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
pipe = pipe.to("cuda")

# 生成图像
prompt = "一只可爱的猫咪在花园里玩耍"
image = pipe(prompt).images[0]
image.save("output.png")
```

## 模型参数

- 精度: {dtype}
- 输入通道: 16 (VAE潜在空间)
- 位置编码: 3轴RoPE
- 归一化: RMSNorm + QK Norm

## 许可证

Apache 2.0

## 引用

如果您使用此模型，请引用Newbie AI项目。
"""

    with open(output_path / "README.md", "w") as f:
        f.write(readme_content)

    print(f"\n✅ README已创建")

    print(f"\n{'='*60}")
    print("🎉 转换完成！")
    print(f"{'='*60}")
    print(f"输出目录: {output_dir}")
    print(f"\n目录结构:")
    print(f"  {output_dir}/")
    print(f"  ├── transformer/")
    print(f"  │   ├── diffusion_pytorch_model.safetensors")
    print(f"  │   └── config.json")
    print(f"  ├── text_encoder/")
    print(f"  │   ├── model.safetensors")
    print(f"  │   ├── config.json")
    print(f"  │   └── tokenizer files...")
    print(f"  ├── clip_model/")
    print(f"  │   ├── model.safetensors")
    print(f"  │   ├── config.json")
    print(f"  │   └── tokenizer files...")
    print(f"  ├── vae/")
    print(f"  │   ├── diffusion_pytorch_model.safetensors")
    print(f"  │   └── config.json")
    print(f"  ├── model_index.json")
    print(f"  └── README.md")

    return output_path


def push_to_hub(output_dir, repo_id, hf_token):
    """推送到HuggingFace Hub"""
    print(f"\n{'='*60}")
    print("☁️  推送到HuggingFace Hub")
    print(f"{'='*60}")
    print(f"Repo ID: {repo_id}")

    try:
        from huggingface_hub import HfApi, create_repo

        api = HfApi()

        # 创建仓库（如果不存在）
        try:
            create_repo(repo_id, token=hf_token, exist_ok=True)
            print(f"✅ 仓库已创建/确认: {repo_id}")
        except Exception as e:
            print(f"⚠️  创建仓库时出现警告: {e}")

        # 上传文件
        print("📤 上传文件中...")
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token
        )

        print(f"✅ 成功推送到: https://huggingface.co/{repo_id}")

    except ImportError:
        print("❌ 需要安装huggingface_hub: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ 推送失败: {e}")
        return False

    return True


def main():
    args = parse_args()

    print("=" * 60)
    print("🚀 Newbie (3B CLIP) 转 Diffusers 格式")
    print("=" * 60)

    # 确定输出目录
    if args.output is None:
        checkpoint_dir = Path(args.checkpoint).parent
        output_dir = checkpoint_dir / "diffusers"
    else:
        output_dir = Path(args.output)

    # 获取dtype
    dtype = get_dtype(args.dtype)
    print(f"\n配置:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Gemma3: {args.gemma3}")
    print(f"  Jina CLIP: {args.jina}")
    print(f"  VAE: {args.vae}")
    print(f"  精度: {args.dtype}")
    print(f"  输出目录: {output_dir}")

    # 1. 加载checkpoint
    state_dict = load_checkpoint(args.checkpoint)

    # 2. 加载文本编码器
    text_encoder, tokenizer, cap_feat_dim = load_text_encoder(
        args.gemma3, dtype, args.hf_token
    )

    # 3. 加载CLIP模型
    clip_model, clip_tokenizer, clip_config = load_clip_model(
        args.jina, dtype, args.hf_token
    )

    # 4. 加载VAE
    vae = load_vae(args.vae, dtype, args.hf_token)

    # 5. 创建DiT模型
    model = create_model(args.model_name, cap_feat_dim)

    # 6. 加载权重
    print(f"\n{'='*60}")
    print("⚙️  加载模型权重")
    print(f"{'='*60}")

    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ 权重加载成功 (strict=True)")
    except Exception as e:
        print(f"⚠️  严格加载失败，尝试非严格加载...")
        print(f"错误: {e}")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"⚠️  缺失的键: {len(missing_keys)}")
        print(f"⚠️  意外的键: {len(unexpected_keys)}")
        if missing_keys:
            print(f"缺失键示例: {missing_keys[:5]}")
        if unexpected_keys:
            print(f"意外键示例: {unexpected_keys[:5]}")

    # 转换为指定精度
    model = model.to(dtype)

    # 7. 转换为Diffusers格式
    output_path = convert_to_diffusers(
        model=model,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        clip_model=clip_model,
        clip_tokenizer=clip_tokenizer,
        vae=vae,
        output_dir=output_dir,
        dtype=args.dtype
    )

    # 8. 推送到Hub（如果需要）
    if args.push_to_hub:
        if args.hub_repo_id is None:
            print("\n❌ 错误: 需要指定 --hub_repo_id 才能推送到Hub")
        else:
            push_to_hub(output_path, args.hub_repo_id, args.hf_token)

    print(f"\n{'='*60}")
    print("✨ 全部完成！")
    print(f"{'='*60}")
    print(f"\n下一步:")
    print(f"1. 测试转换后的模型:")
    print(f"   cd {output_path}")
    print(f"   python -c \"from diffusers import DiffusionPipeline; pipe = DiffusionPipeline.from_pretrained('.', trust_remote_code=True)\"")
    print(f"\n2. 推送到HuggingFace Hub (可选):")
    print(f"   python {__file__} --checkpoint {args.checkpoint} --push_to_hub --hub_repo_id your-username/newbie-diffusers")


if __name__ == "__main__":
    main()

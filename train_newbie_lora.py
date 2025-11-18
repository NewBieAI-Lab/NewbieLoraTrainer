#!/usr/bin/env python3
"""Newbie LoRA 训练器 - 基于 Rectified Flow 的 LoRA 微调"""

import os
import sys
import argparse
import toml
import json
import logging
import random
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed, DummyOptim, DummyScheduler
from diffusers import AutoencoderKL, DiT2D, DiT3D, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent))
import models
from transport import create_transport

try:
    import bitsandbytes as bnb
except ImportError:
    logging.warning("bitsandbytes not available, 8-bit optimizer disabled")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("newbie_lora_trainer")


class ImageCaptionDataset(Dataset):
    """图像-文本对数据集，支持 kohya_ss 风格目录重复"""

    def __init__(self, train_data_dir: str, resolution: int, enable_bucket: bool = True):
        self.train_data_dir = train_data_dir
        self.resolution = resolution
        self.enable_bucket = enable_bucket
        self.image_paths = []
        self.captions = []
        self.repeats = []
        self.buckets = {}
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

        self._load_data()
        if self.enable_bucket:
            self._prepare_buckets()

    def _load_data(self):
        logger.info(f"Loading data from: {self.train_data_dir}")
        
        for root, _, files in os.walk(self.train_data_dir):
            dir_name = os.path.basename(root)
            repeats = int(dir_name.split('_')[0]) if '_' in dir_name and dir_name[0].isdigit() else 1

            for file in files:
                _, ext = os.path.splitext(file)
                if ext.lower() in self.image_extensions:
                    image_path = os.path.join(root, file)
                    caption_path = os.path.splitext(image_path)[0] + '.txt'

                    caption = ''
                    if os.path.exists(caption_path):
                        try:
                            with open(caption_path, 'r', encoding='utf-8') as f:
                                caption = f.read().strip()
                        except UnicodeDecodeError:
                            with open(caption_path, 'r', encoding='latin-1') as f:
                                caption = f.read().strip()

                    self.image_paths.append(image_path)
                    self.captions.append(caption)
                    self.repeats.append(repeats)

        logger.info(f"Loaded {len(self.image_paths)} image-caption pairs")
    
    def _prepare_buckets(self):
        """创建多宽高比分辨率桶"""
        aspect_ratios = [(1, 1), (3, 4), (4, 3), (9, 16), (16, 9)]
        buckets = []

        for ar in aspect_ratios:
            width = int(self.resolution * ar[0] / max(ar))
            height = int(self.resolution * ar[1] / max(ar))
            width = (width // 64) * 64  # 对齐64
            height = (height // 64) * 64
            buckets.append((width, height))

        buckets = list(set(buckets))
        buckets.sort(key=lambda x: x[0] * x[1])

        for bucket in buckets:
            self.buckets[bucket] = []

        logger.info(f"Created {len(buckets)} resolution buckets: {buckets}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image

        image_path = self.image_paths[idx]
        caption = self.captions[idx]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load {image_path}: {e}")
            image = Image.new("RGB", (self.resolution, self.resolution), color="black")

        if self.enable_bucket:
            orig_width, orig_height = image.size
            orig_ratio = orig_width / orig_height
            closest_bucket = min(self.buckets.keys(), key=lambda x: abs((x[0] / x[1]) - orig_ratio))
            target_width, target_height = closest_bucket
        else:
            target_width = target_height = self.resolution

        transform = transforms.Compose([
            transforms.Resize((target_height, target_width), interpolation=InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        return {"pixel_values": transform(image), "caption": caption}


def collate_fn(batch):
    """数据批次处理函数"""
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    captions = [example["caption"] for example in batch]
    
    return {
        "pixel_values": pixel_values,
        "captions": captions
    }


def load_model_and_tokenizer(config):
    """
    加载模型和分词器
    支持两种模式：1) Diffusers格式自动检测 2) 分别指定路径
    """
    base_model_path = config['Model']['base_model_path']
    trust_remote_code = config['Model'].get('trust_remote_code', True)
    model_index_path = os.path.join(base_model_path, "model_index.json")
    is_diffusers_format = os.path.exists(model_index_path)

    logger.info(f"Loading model from: {base_model_path}")

    if is_diffusers_format:
        logger.info("Diffusers format detected, auto-loading components")

        text_encoder_path = os.path.join(base_model_path, "text_encoder")
        text_encoder = AutoModel.from_pretrained(text_encoder_path, torch_dtype=torch.float32, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, trust_remote_code=trust_remote_code)
        tokenizer.padding_side = "right"

        clip_model_path = os.path.join(base_model_path, "clip_model")
        clip_model = AutoModel.from_pretrained(clip_model_path, torch_dtype=torch.float32, trust_remote_code=True)
        clip_tokenizer = AutoTokenizer.from_pretrained(clip_model_path, trust_remote_code=True)

        transformer_path = os.path.join(base_model_path, "transformer")
        config_path = os.path.join(transformer_path, "config.json")

        with open(config_path, 'r') as f:
            model_config = json.load(f)

        cap_feat_dim = text_encoder.config.text_config.hidden_size
        model_name = model_config.get('_class_name', 'NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP')

        model = models.__dict__[model_name](
            in_channels=model_config.get('in_channels', 16),
            qk_norm=True,
            cap_feat_dim=cap_feat_dim,
            clip_text_dim=model_config.get('clip_text_dim', 1024),
            clip_img_dim=model_config.get('clip_img_dim', 1024),
        )

        weight_path = os.path.join(transformer_path, "diffusion_pytorch_model.safetensors")
        if os.path.exists(weight_path):
            state_dict = load_file(weight_path)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {len(unexpected_keys)}")

        vae_path = os.path.join(base_model_path, "vae")
        vae = AutoencoderKL.from_pretrained(
            vae_path if os.path.exists(vae_path) else "stabilityai/sdxl-vae",
            torch_dtype=torch.float32,
            trust_remote_code=trust_remote_code
        )

    else:
        logger.info("Loading from separate paths")

        gemma_path = config['Model'].get('gemma_model_path', 'google/gemma-3-4b-it')
        clip_path = config['Model'].get('clip_model_path', 'jinaai/jina-clip-v2')
        transformer_path = config['Model'].get('transformer_path', None)

        text_encoder = AutoModel.from_pretrained(gemma_path, torch_dtype=torch.float32, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(gemma_path, trust_remote_code=trust_remote_code)
        tokenizer.padding_side = "right"

        clip_model = AutoModel.from_pretrained(clip_path, torch_dtype=torch.float32, trust_remote_code=True)
        clip_tokenizer = AutoTokenizer.from_pretrained(clip_path, trust_remote_code=True)

        cap_feat_dim = text_encoder.config.text_config.hidden_size
        model = models.NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP(
            in_channels=16, qk_norm=True, cap_feat_dim=cap_feat_dim,
            clip_text_dim=1024, clip_img_dim=1024
        )

        if transformer_path and os.path.exists(transformer_path):
            state_dict = load_file(transformer_path) if transformer_path.endswith('.safetensors') else torch.load(transformer_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)

        vae_path = config['Model'].get('vae_path', 'stabilityai/sdxl-vae')
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float32, trust_remote_code=trust_remote_code)

    model.train()
    vae.eval()
    text_encoder.eval()
    clip_model.eval()

    logger.info(f"Model loaded: {model.parameter_count():,} params, cap_feat_dim={cap_feat_dim}")

    return model, vae, text_encoder, tokenizer, clip_model, clip_tokenizer


def setup_lora(model, config):
    """应用 LoRA 到模型"""
    lora_rank = config['Model']['lora_rank']
    lora_alpha = config['Model']['lora_alpha']
    lora_target_modules = config['Model']['lora_target_modules']
    lora_dropout = config['Model'].get('lora_dropout', 0.05)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none"
    )

    peft_model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    logger.info(f"LoRA applied: {trainable_params/1e6:.2f}M/{total_params/1e6:.2f}M trainable ({trainable_params/total_params*100:.2f}%)")
    logger.info(f"  Target modules: {lora_target_modules}")

    return peft_model


def setup_optimizer(model, config):
    """设置优化器 (LoRA优化参数)"""
    optimizer_type = config['Optimization']['optimizer_type']
    learning_rate = config['Model']['learning_rate']
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    adam_kwargs = {"lr": learning_rate, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}

    if optimizer_type == "AdamW8bit":
        try:
            optimizer = bnb.optim.AdamW8bit(trainable_params, **adam_kwargs)
            logger.info("Using 8-bit AdamW optimizer")
        except Exception as e:
            logger.warning(f"8-bit AdamW failed, using standard AdamW: {e}")
            optimizer = optim.AdamW(trainable_params, **adam_kwargs)
    else:
        optimizer = optim.AdamW(trainable_params, **adam_kwargs)
        logger.info("Using standard AdamW optimizer")

    return optimizer


def setup_scheduler(optimizer, config, train_dataloader):
    """设置学习率调度器"""
    num_epochs = config['Model']['num_epochs']
    num_training_steps = num_epochs * len(train_dataloader)
    scheduler_type = config['Model']['lr_scheduler']
    lr_warmup_steps = config['Model'].get('lr_warmup_steps', 100)

    scheduler = get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=num_training_steps
    )

    logger.info(f"LR scheduler: {scheduler_type}, {num_training_steps} steps, {lr_warmup_steps} warmup")

    return scheduler, num_training_steps


def generate_noise(batch_size, num_channels, height, width, device):
    """生成随机噪声"""
    return torch.randn((batch_size, num_channels, height, width), device=device)


def compute_loss(model, vae, text_encoder, tokenizer, clip_model, clip_tokenizer, transport, batch, device):
    """计算 Rectified Flow 训练损失"""
    pixel_values = batch["pixel_values"].to(device)
    captions = batch["captions"]
    batch_size = pixel_values.shape[0]

    with torch.no_grad():
        # Gemma3: 主文本特征 (max_length=512, 使用倒数第二层)
        gemma_inputs = tokenizer(
            captions, padding=True, pad_to_multiple_of=8,
            truncation=True, max_length=512, return_tensors="pt"
        ).to(device)
        gemma_outputs = text_encoder(**gemma_inputs, output_hidden_states=True)
        cap_feats = gemma_outputs.hidden_states[-2]
        cap_mask = gemma_inputs.attention_mask

        # Jina CLIP: 池化文本特征 (max_length=2048)
        clip_inputs = clip_tokenizer(
            captions, padding=True, truncation=True,
            max_length=2048, return_tensors="pt"
        ).to(device)
        clip_text_pooled = clip_model.get_text_features(**clip_inputs)

        # VAE 编码
        latents = vae.encode(pixel_values).latent_dist.sample()
        scaling_factor = getattr(vae.config, 'scaling_factor', 0.13025)
        latents = latents * scaling_factor

    latents_list = [latents[i] for i in range(batch_size)]
    model_kwargs = dict(cap_feats=cap_feats, cap_mask=cap_mask, clip_text_pooled=clip_text_pooled)

    # Rectified Flow 损失计算
    loss_dict = transport.training_losses(model, latents_list, model_kwargs)
    return loss_dict["loss"].mean()


def save_checkpoint(accelerator, model, optimizer, scheduler, step, config):
    """保存训练检查点"""
    checkpoint_dir = os.path.join(config['Model']['output_dir'], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")

    checkpoint = {
        "step": step,
        "lora_state_dict": get_peft_model_state_dict(model),
        "optimizer_state_dict": accelerator.get_state_dict(optimizer),
        "scheduler_state_dict": scheduler.state_dict()
    }

    accelerator.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    save_lora_model(accelerator, model, config, step)


def save_lora_model(accelerator, model, config, step=None):
    """保存 LoRA 模型"""
    output_dir = config['Model']['output_dir']
    output_name = config['Model']['output_name']
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, f"{output_name}_step_{step}" if step else output_name)

    accelerator.unwrap_model(model).save_pretrained(
        save_path,
        is_main_process=accelerator.is_main_process,
        state_dict=accelerator.get_state_dict(model)
    )

    if accelerator.is_main_process:
        logger.info(f"LoRA model saved: {save_path}")


def load_checkpoint(accelerator, model, optimizer, scheduler, config):
    """从检查点加载训练状态"""
    checkpoint_dir = os.path.join(config['Model']['output_dir'], "checkpoints")
    if not os.path.exists(checkpoint_dir):
        logger.info("No checkpoint found, starting from scratch")
        return 0

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pt")]
    if not checkpoints:
        logger.info("No checkpoint files found, starting from scratch")
        return 0

    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])

    logger.info(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    set_peft_model_state_dict(model, checkpoint["lora_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info(f"Resumed from step {checkpoint['step']}")
    return checkpoint["step"]


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description="Newbie LoRA Trainer")
    parser.add_argument("--config_file", type=str, required=True, help="Path to .toml config file")
    args = parser.parse_args()

    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = toml.load(f)

    output_dir = config['Model']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=config['Model'].get('mixed_precision', 'no'),
        log_with="tensorboard",
        project_dir=output_dir,
        kwargs_handlers=[ddp_kwargs]
    )

    set_seed(42)

    if not config['Optimization'].get('use_flash_attention_2', False):
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)

    model, vae, text_encoder, tokenizer, clip_model, clip_tokenizer = load_model_and_tokenizer(config)

    # 创建 Rectified Flow transport
    resolution = config['Model']['resolution']
    seq_len = (resolution // 16) ** 2
    transport = create_transport(
        path_type="Linear",
        prediction="velocity",
        snr_type="lognorm",
        do_shift=True,
        seq_len=seq_len
    )
    logger.info(f"Rectified Flow transport created (resolution={resolution}, seq_len={seq_len})")

    model = setup_lora(model, config)

    if config['Model'].get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    dataset = ImageCaptionDataset(
        train_data_dir=config['Model']['train_data_dir'],
        resolution=resolution,
        enable_bucket=config['Model'].get('enable_bucket', True)
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=config['Model']['train_batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    optimizer = setup_optimizer(model, config)
    scheduler, num_training_steps = setup_scheduler(optimizer, config, train_dataloader)

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)
    vae, text_encoder, clip_model = accelerator.prepare(vae), accelerator.prepare(text_encoder), accelerator.prepare(clip_model)

    start_step = load_checkpoint(accelerator, model, optimizer, scheduler, config)

    logger.info("Training started")
    global_step = start_step

    if accelerator.is_main_process:
        accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])

    start_time = datetime.now()
    max_grad_norm = config['Optimization'].get('gradient_clip_norm', 1.0)

    for epoch in range(config['Model']['num_epochs']):
        for batch in train_dataloader:
            global_step += 1

            loss = compute_loss(model, vae, text_encoder, tokenizer, clip_model, clip_tokenizer, transport, batch, accelerator.device)

            accelerator.backward(loss)

            if max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if accelerator.is_main_process:
                accelerator.log({"loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0]}, step=global_step)

            if global_step % 100 == 0 or global_step == 1:
                elapsed = datetime.now() - start_time
                steps_per_sec = global_step / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
                logger.info(f"Epoch {epoch+1}/{config['Model']['num_epochs']}, Step {global_step}/{num_training_steps}, Loss {loss.item():.4f}, LR {scheduler.get_last_lr()[0]:.7f}, Speed {steps_per_sec:.2f} steps/s")

            if global_step % 1000 == 0:
                save_checkpoint(accelerator, model, optimizer, scheduler, global_step, config)

        save_checkpoint(accelerator, model, optimizer, scheduler, global_step, config)

    logger.info("Training complete, saving final model")
    save_lora_model(accelerator, model, config)

    if accelerator.is_main_process:
        accelerator.end_training()

    logger.info("Training finished")


if __name__ == "__main__":
    main()
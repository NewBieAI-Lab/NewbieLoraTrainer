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
from accelerate.utils import set_seed
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from tqdm import tqdm

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

    def __init__(
        self,
        train_data_dir: str,
        resolution: int,
        enable_bucket: bool = True,
        use_cache: bool = True,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        clip_model=None,
        clip_tokenizer=None,
        device=None,
        dtype=torch.bfloat16,
        gemma3_prompt: str = "",
        min_bucket_reso: int = 256,
        max_bucket_reso: int = 2048,
        bucket_reso_step: int = 64,
        cache_refined_cap_feats: bool = False,
        max_token_length: int = 512,
    ):
        self.train_data_dir = train_data_dir
        self.resolution = resolution
        self.enable_bucket = enable_bucket
        self.use_cache = use_cache
        self.image_paths = []
        self.captions = []
        self.repeats = []
        self.image_resolutions = []
        self.buckets = {}
        self.bucket_resolutions = []
        self.image_to_bucket = {}
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.device = device
        self.dtype = dtype
        self.gemma3_prompt = gemma3_prompt
        self.min_bucket_reso = min_bucket_reso
        self.max_bucket_reso = max_bucket_reso
        self.bucket_reso_step = bucket_reso_step
        self.cache_refined_cap_feats = cache_refined_cap_feats
        self.max_token_length = max_token_length

        self._load_data()
        if self.enable_bucket:
            self._generate_buckets()
            self._assign_buckets()
        else:
            for idx in range(len(self.image_paths)):
                self.image_to_bucket[idx] = (self.resolution, self.resolution)

        if self.use_cache and vae is not None:
            self._prepare_cache()

    def _load_data(self):
        from PIL import Image
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

                    try:
                        with Image.open(image_path) as img:
                            width, height = img.size
                            self.image_resolutions.append((width, height))
                    except Exception as e:
                        logger.warning(f"Could not read image size for {image_path}: {e}")
                        self.image_resolutions.append((self.resolution, self.resolution))

                    self.image_paths.append(image_path)
                    self.captions.append(caption)
                    self.repeats.append(repeats)

        logger.info(f"Loaded {len(self.image_paths)} unique images")

    def _prepare_cache(self):
        from PIL import Image
        from tqdm import tqdm
        try:
            import cv2
            import numpy as np
            HAS_CV2 = True
        except ImportError:
            HAS_CV2 = False
            logger.warning("OpenCV not available, WebP support may be limited")

        logger.info("Checking cache files...")
        missing_indices = []

        for idx, image_path in enumerate(self.image_paths):
            vae_cache = f"{image_path}.safetensors"
            text_cache = f"{os.path.splitext(image_path)[0]}.txt.safetensors"

            if not os.path.exists(vae_cache) or not os.path.exists(text_cache):
                missing_indices.append(idx)

        if missing_indices:
            logger.info(f"Generating {len(missing_indices)} cache files...")
            self.vae.eval().to(self.device)
            self.text_encoder.eval().to(self.device)
            self.clip_model.eval().to(self.device)

            context_refiner = None
            cap_embedder = None
            rope_embedder = None
            if self.cache_refined_cap_feats:
                logger.info("Loading cap_embedder and context_refiner for refined cap_feats caching...")
                from models.model import RopeEmbedder
                cap_feat_dim = self.text_encoder.config.text_config.hidden_size
                temp_model = models.NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP(
                    in_channels=16, qk_norm=True, cap_feat_dim=cap_feat_dim,
                    clip_text_dim=1024, clip_img_dim=1024
                )
                cap_embedder = temp_model.cap_embedder.eval().to(self.device)
                context_refiner = temp_model.context_refiner.eval().to(self.device)
                rope_embedder = RopeEmbedder(axes_dims=[32, 32, 32], axes_lens=[1024, 512, 512])

            with torch.no_grad():
                for idx in tqdm(missing_indices, desc="Caching"):
                    image_path = self.image_paths[idx]
                    caption = self.captions[idx]
                    vae_cache = f"{image_path}.safetensors"
                    text_cache = f"{os.path.splitext(image_path)[0]}.txt.safetensors"

                    try:
                        try:
                            image = Image.open(image_path).convert("RGB")
                        except Exception as e:
                            if HAS_CV2:
                                logger.warning(f"PIL failed for {image_path}, using OpenCV")
                                img_array = cv2.imread(image_path, cv2.IMREAD_COLOR)
                                if img_array is None:
                                    raise ValueError(f"Cannot load: {image_path}")
                                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                                image = Image.fromarray(img_array)
                            else:
                                raise e

                        bucket_reso = self.image_to_bucket.get(idx, (self.resolution, self.resolution))
                        target_width, target_height = bucket_reso

                        bucket_ratio = target_width / target_height
                        orig_ratio = image.size[0] / image.size[1]

                        if orig_ratio > bucket_ratio:
                            resize_height = target_height
                            resize_width = int(resize_height * orig_ratio)
                        else:
                            resize_width = target_width
                            resize_height = int(resize_width / orig_ratio)

                        transform = transforms.Compose([
                            transforms.Resize((resize_height, resize_width), interpolation=InterpolationMode.LANCZOS),
                            transforms.CenterCrop((target_height, target_width)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])
                        ])

                        pixel_values = transform(image).unsqueeze(0).to(self.device)

                        latents = self.vae.encode(pixel_values).latent_dist.mode()
                        scaling_factor = getattr(self.vae.config, 'scaling_factor', 0.13025)
                        latents = (latents * scaling_factor).squeeze(0).cpu()

                        save_file({
                            "latents": latents,
                            "width": torch.tensor(target_width),
                            "height": torch.tensor(target_height)
                        }, vae_cache)

                        with torch.autocast(device_type='cuda', dtype=self.dtype):
                            gemma_text = self.gemma3_prompt + caption if self.gemma3_prompt else caption
                            gemma_inputs = self.tokenizer(
                                [gemma_text], padding=True, pad_to_multiple_of=8,
                                truncation=True, max_length=self.max_token_length, return_tensors="pt"
                            ).to(self.device)
                            gemma_outputs = self.text_encoder(**gemma_inputs, output_hidden_states=True)
                            cap_feats = gemma_outputs.hidden_states[-2].to(dtype=self.dtype)
                            cap_mask = gemma_inputs.attention_mask

                            if self.cache_refined_cap_feats:
                                actual_len = cap_feats.shape[1]

                                cap_feats = cap_embedder(cap_feats)

                                if actual_len < self.max_token_length:
                                    pad_len = self.max_token_length - actual_len
                                    cap_feats = torch.cat([
                                        cap_feats,
                                        torch.zeros(1, pad_len, cap_feats.shape[-1], dtype=cap_feats.dtype, device=cap_feats.device)
                                    ], dim=1)
                                    cap_mask = torch.cat([
                                        cap_mask,
                                        torch.zeros(1, pad_len, dtype=cap_mask.dtype, device=cap_mask.device)
                                    ], dim=1)

                                position_ids = torch.zeros(1, self.max_token_length, 3, dtype=torch.int32, device=self.device)
                                position_ids[0, :actual_len, 0] = torch.arange(actual_len, dtype=torch.int32, device=self.device)
                                cap_freqs_cis = rope_embedder(position_ids)

                                for layer in context_refiner:
                                    cap_feats = layer(cap_feats, cap_mask, cap_freqs_cis)

                            clip_inputs = self.clip_tokenizer(
                                [caption], padding=True, truncation=True,
                                max_length=2048, return_tensors="pt"
                            ).to(self.device)
                            clip_text_pooled = self.clip_model.get_text_features(**clip_inputs).squeeze(0).to(dtype=self.dtype).cpu()

                        if self.cache_refined_cap_feats:
                            save_file({
                                "cap_feats_refined": cap_feats.squeeze(0).cpu(),
                                "cap_mask": cap_mask.squeeze(0).cpu(),
                                "clip_text_pooled": clip_text_pooled
                            }, text_cache)
                        else:
                            save_file({
                                "cap_feats": cap_feats.squeeze(0).cpu(),
                                "cap_mask": cap_mask.squeeze(0).cpu(),
                                "clip_text_pooled": clip_text_pooled
                            }, text_cache)

                    except Exception as e:
                        logger.error(f"Cache error for {image_path}: {e}")

            logger.info("Cache generation complete")
            if self.cache_refined_cap_feats:
                del cap_embedder, context_refiner, rope_embedder
                import gc
                gc.collect()
                torch.cuda.empty_cache()
        else:
            logger.info("All cache files found")

    def _generate_buckets(self):
        max_reso = self.resolution
        max_tokens = (max_reso / 16) * (max_reso / 16)

        assert self.bucket_reso_step % 8 == 0, "bucket_reso_step must be divisible by 8"
        assert self.min_bucket_reso % 8 == 0, "min_bucket_reso must be divisible by 8"
        assert self.max_bucket_reso % 8 == 0, "max_bucket_reso must be divisible by 8"

        resolutions = set()

        w = self.min_bucket_reso
        while w <= self.max_bucket_reso:
            h = self.min_bucket_reso
            while h <= self.max_bucket_reso:
                if (w / 16) * (h / 16) <= max_tokens and w % 8 == 0 and h % 8 == 0:
                    resolutions.add((w, h))
                h += self.bucket_reso_step
            w += self.bucket_reso_step

        self.bucket_resolutions = sorted(list(resolutions), key=lambda x: x[0] / x[1])

        for reso in self.bucket_resolutions:
            self.buckets[reso] = []

        logger.info(f"Generated {len(self.bucket_resolutions)} buckets with max {max_tokens:.0f} tokens")
        logger.info(f"Example buckets: {self.bucket_resolutions[:5]} ... {self.bucket_resolutions[-5:]}")

    def _assign_buckets(self):
        ar_errors = []

        for idx, (img_w, img_h) in enumerate(self.image_resolutions):
            aspect_ratio = img_w / img_h

            closest_bucket = min(
                self.bucket_resolutions,
                key=lambda x: abs((x[0] / x[1]) - aspect_ratio)
            )

            bucket_ar = closest_bucket[0] / closest_bucket[1]
            ar_error = abs(bucket_ar - aspect_ratio)
            ar_errors.append(ar_error)

            self.buckets[closest_bucket].append(idx)
            self.image_to_bucket[idx] = closest_bucket

        bucket_counts_with_repeats = {}
        for bucket_reso, indices in self.buckets.items():
            if len(indices) > 0:
                total_with_repeats = sum(self.repeats[idx] for idx in indices)
                bucket_counts_with_repeats[bucket_reso] = total_with_repeats

        logger.info(f"Bucket assignment (with repeats): {bucket_counts_with_repeats}")

        if ar_errors:
            import numpy as np
            ar_errors_array = np.array(ar_errors)
            logger.info(f"Mean AR error: {np.mean(ar_errors_array):.4f}")
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = self.captions[idx]
        bucket_reso = self.image_to_bucket.get(idx, (self.resolution, self.resolution))

        if self.use_cache:
            vae_cache = f"{image_path}.safetensors"
            text_cache = f"{os.path.splitext(image_path)[0]}.txt.safetensors"

            vae_data = load_file(vae_cache)
            text_data = load_file(text_cache)

            if self.cache_refined_cap_feats:
                return {
                    "latents": vae_data['latents'],
                    "cap_feats": text_data['cap_feats_refined'],
                    "cap_mask": text_data['cap_mask'],
                    "clip_text_pooled": text_data['clip_text_pooled'],
                    "cached": True,
                    "refined": True,
                }
            else:
                return {
                    "latents": vae_data['latents'],
                    "cap_feats": text_data['cap_feats'],
                    "cap_mask": text_data['cap_mask'],
                    "clip_text_pooled": text_data['clip_text_pooled'],
                    "cached": True,
                    "refined": False,
                }
        else:
            from PIL import Image
            try:
                import cv2
                import numpy as np
                HAS_CV2 = True
            except ImportError:
                HAS_CV2 = False

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                if HAS_CV2:
                    try:
                        logger.warning(f"PIL failed for {image_path}, trying OpenCV")
                        img_array = cv2.imread(image_path, cv2.IMREAD_COLOR)
                        if img_array is not None:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                            image = Image.fromarray(img_array)
                        else:
                            raise ValueError(f"OpenCV also failed to load {image_path}")
                    except Exception as cv_err:
                        logger.error(f"Both PIL and OpenCV failed for {image_path}: {e}, {cv_err}")
                        image = Image.new("RGB", (self.resolution, self.resolution), color="black")
                else:
                    logger.error(f"Failed to load {image_path}: {e}")
                    image = Image.new("RGB", (self.resolution, self.resolution), color="black")

            target_width, target_height = bucket_reso

            bucket_ratio = target_width / target_height
            orig_ratio = image.size[0] / image.size[1]

            if orig_ratio > bucket_ratio:
                resize_height = target_height
                resize_width = int(resize_height * orig_ratio)
            else:
                resize_width = target_width
                resize_height = int(resize_width / orig_ratio)

            transform = transforms.Compose([
                transforms.Resize((resize_height, resize_width), interpolation=InterpolationMode.LANCZOS),
                transforms.CenterCrop((target_height, target_width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            return {"pixel_values": transform(image), "caption": caption, "cached": False}


class BucketBatchSampler:
    def __init__(self, dataset, batch_size, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.bucket_to_indices = {}
        for idx in range(len(dataset)):
            bucket_reso = dataset.image_to_bucket.get(idx, (dataset.resolution, dataset.resolution))
            if bucket_reso not in self.bucket_to_indices:
                self.bucket_to_indices[bucket_reso] = []

            for _ in range(dataset.repeats[idx]):
                self.bucket_to_indices[bucket_reso].append(idx)

        total_batches = sum(math.ceil(len(indices) / batch_size) for indices in self.bucket_to_indices.values())
        logger.info(f"BucketBatchSampler: {len(self.bucket_to_indices)} buckets, {total_batches} batches")

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        bucket_batches = []
        for bucket_reso, indices in self.bucket_to_indices.items():
            indices_copy = indices.copy()
            if self.shuffle:
                rng.shuffle(indices_copy)

            batch_count = math.ceil(len(indices_copy) / self.batch_size)
            for batch_index in range(batch_count):
                start_idx = batch_index * self.batch_size
                batch = indices_copy[start_idx:start_idx + self.batch_size]
                bucket_batches.append(batch)

        if self.shuffle:
            rng.shuffle(bucket_batches)

        for batch in bucket_batches:
            yield batch

    def __len__(self):
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.bucket_to_indices.values())

    def set_epoch(self, epoch):
        self.epoch = epoch


def collate_fn(batch):
    if batch[0].get("cached", False):
        if batch[0].get("refined", False):
            return {
                "latents": torch.stack([example["latents"] for example in batch]),
                "cap_feats": torch.stack([example["cap_feats"] for example in batch]),
                "cap_mask": torch.stack([example["cap_mask"] for example in batch]),
                "clip_text_pooled": torch.stack([example["clip_text_pooled"] for example in batch]),
                "cached": True,
                "refined": True,
            }
        else:
            max_cap_len = max(example["cap_feats"].shape[0] for example in batch)

            cap_feats_list = []
            cap_mask_list = []
            for example in batch:
                cap_feat = example["cap_feats"]
                cap_mask = example["cap_mask"]
                current_len = cap_feat.shape[0]

                if current_len < max_cap_len:
                    pad_len = max_cap_len - current_len
                    cap_feat = torch.cat([cap_feat, torch.zeros(pad_len, cap_feat.shape[1], dtype=cap_feat.dtype)], dim=0)
                    cap_mask = torch.cat([cap_mask, torch.zeros(pad_len, dtype=cap_mask.dtype)], dim=0)

                cap_feats_list.append(cap_feat)
                cap_mask_list.append(cap_mask)

            return {
                "latents": torch.stack([example["latents"] for example in batch]),
                "cap_feats": torch.stack(cap_feats_list),
                "cap_mask": torch.stack(cap_mask_list),
                "clip_text_pooled": torch.stack([example["clip_text_pooled"] for example in batch]),
                "cached": True,
                "refined": False,
            }
    else:
        return {
            "pixel_values": torch.stack([example["pixel_values"] for example in batch]),
            "captions": [example["caption"] for example in batch],
            "cached": False,
        }


def load_encoders_only(config):
    base_model_path = config['Model']['base_model_path']
    trust_remote_code = config['Model'].get('trust_remote_code', True)
    model_index_path = os.path.join(base_model_path, "model_index.json")
    is_diffusers_format = os.path.exists(model_index_path)

    mixed_precision = config['Model'].get('mixed_precision', 'no')
    if mixed_precision == 'bf16':
        model_dtype = torch.bfloat16
    elif mixed_precision == 'fp16':
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    logger.info("Loading encoders only (VAE, text_encoder, CLIP) for cache generation...")

    if is_diffusers_format:
        text_encoder_path = os.path.join(base_model_path, "text_encoder")
        text_encoder = AutoModel.from_pretrained(text_encoder_path, torch_dtype=model_dtype, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, trust_remote_code=trust_remote_code)
        tokenizer.padding_side = "right"

        clip_model_path = os.path.join(base_model_path, "clip_model")
        clip_model = AutoModel.from_pretrained(clip_model_path, torch_dtype=model_dtype, trust_remote_code=True)
        clip_tokenizer = AutoTokenizer.from_pretrained(clip_model_path, trust_remote_code=True)

        vae_path = os.path.join(base_model_path, "vae")
        vae = AutoencoderKL.from_pretrained(
            vae_path if os.path.exists(vae_path) else "stabilityai/sdxl-vae",
            torch_dtype=torch.float32,
            trust_remote_code=trust_remote_code
        )
    else:
        gemma_path = config['Model'].get('gemma_model_path', 'google/gemma-3-4b-it')
        clip_path = config['Model'].get('clip_model_path', 'jinaai/jina-clip-v2')

        text_encoder = AutoModel.from_pretrained(gemma_path, torch_dtype=model_dtype, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(gemma_path, trust_remote_code=trust_remote_code)
        tokenizer.padding_side = "right"

        clip_model = AutoModel.from_pretrained(clip_path, torch_dtype=model_dtype, trust_remote_code=True)
        clip_tokenizer = AutoTokenizer.from_pretrained(clip_path, trust_remote_code=True)

        vae_path = config['Model'].get('vae_path', 'stabilityai/sdxl-vae')
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float32, trust_remote_code=trust_remote_code)

    vae.eval()
    vae.requires_grad_(False)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    clip_model.eval()
    clip_model.requires_grad_(False)

    logger.info("Encoders loaded successfully")
    return vae, text_encoder, tokenizer, clip_model, clip_tokenizer


def load_transformer_only(config):
    base_model_path = config['Model']['base_model_path']
    trust_remote_code = config['Model'].get('trust_remote_code', True)
    model_index_path = os.path.join(base_model_path, "model_index.json")
    is_diffusers_format = os.path.exists(model_index_path)

    mixed_precision = config['Model'].get('mixed_precision', 'no')
    if mixed_precision == 'bf16':
        model_dtype = torch.bfloat16
    elif mixed_precision == 'fp16':
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    logger.info(f"Loading transformer only from: {base_model_path}")

    if is_diffusers_format:
        transformer_path = os.path.join(base_model_path, "transformer")
        config_path = os.path.join(transformer_path, "config.json")

        with open(config_path, 'r') as f:
            model_config = json.load(f)

        text_encoder_path = os.path.join(base_model_path, "text_encoder")
        text_encoder_config = AutoConfig.from_pretrained(text_encoder_path, trust_remote_code=trust_remote_code)
        cap_feat_dim = text_encoder_config.text_config.hidden_size

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

        if model_dtype != torch.float32:
            model = model.to(dtype=model_dtype)
            logger.info(f"Model converted to {model_dtype}")

    else:
        gemma_path = config['Model'].get('gemma_model_path', 'google/gemma-3-4b-it')

        text_encoder_config = AutoConfig.from_pretrained(gemma_path, trust_remote_code=trust_remote_code)
        cap_feat_dim = text_encoder_config.text_config.hidden_size

        transformer_path = config['Model'].get('transformer_path', None)

        model = models.NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP(
            in_channels=16, qk_norm=True, cap_feat_dim=cap_feat_dim,
            clip_text_dim=1024, clip_img_dim=1024
        )

        if transformer_path and os.path.exists(transformer_path):
            state_dict = load_file(transformer_path) if transformer_path.endswith('.safetensors') else torch.load(transformer_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)

        if model_dtype != torch.float32:
            model = model.to(dtype=model_dtype)
            logger.info(f"Model converted to {model_dtype}")

    model.train()
    logger.info(f"Transformer loaded: {model.parameter_count():,} params, cap_feat_dim={cap_feat_dim}")

    return model


def load_model_and_tokenizer(config):
    base_model_path = config['Model']['base_model_path']
    trust_remote_code = config['Model'].get('trust_remote_code', True)
    model_index_path = os.path.join(base_model_path, "model_index.json")
    is_diffusers_format = os.path.exists(model_index_path)

    mixed_precision = config['Model'].get('mixed_precision', 'no')
    if mixed_precision == 'bf16':
        model_dtype = torch.bfloat16
    elif mixed_precision == 'fp16':
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    logger.info(f"Loading model from: {base_model_path}")

    if is_diffusers_format:
        logger.info("Diffusers format detected, auto-loading components")

        text_encoder_path = os.path.join(base_model_path, "text_encoder")
        text_encoder = AutoModel.from_pretrained(text_encoder_path, torch_dtype=model_dtype, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, trust_remote_code=trust_remote_code)
        tokenizer.padding_side = "right"

        clip_model_path = os.path.join(base_model_path, "clip_model")
        clip_model = AutoModel.from_pretrained(clip_model_path, torch_dtype=model_dtype, trust_remote_code=True)
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

        # Convert model to target dtype
        if model_dtype != torch.float32:
            model = model.to(dtype=model_dtype)
            logger.info(f"Model converted to {model_dtype}")

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

        text_encoder = AutoModel.from_pretrained(gemma_path, torch_dtype=model_dtype, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(gemma_path, trust_remote_code=trust_remote_code)
        tokenizer.padding_side = "right"

        clip_model = AutoModel.from_pretrained(clip_path, torch_dtype=model_dtype, trust_remote_code=True)
        clip_tokenizer = AutoTokenizer.from_pretrained(clip_path, trust_remote_code=True)

        cap_feat_dim = text_encoder.config.text_config.hidden_size
        model = models.NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP(
            in_channels=16, qk_norm=True, cap_feat_dim=cap_feat_dim,
            clip_text_dim=1024, clip_img_dim=1024
        )

        if transformer_path and os.path.exists(transformer_path):
            state_dict = load_file(transformer_path) if transformer_path.endswith('.safetensors') else torch.load(transformer_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)

        # Convert model to target dtype
        if model_dtype != torch.float32:
            model = model.to(dtype=model_dtype)
            logger.info(f"Model converted to {model_dtype}")

        vae_path = config['Model'].get('vae_path', 'stabilityai/sdxl-vae')
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float32, trust_remote_code=trust_remote_code)

    model.train()
    vae.eval()
    vae.requires_grad_(False)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    clip_model.eval()
    clip_model.requires_grad_(False)

    logger.info(f"Model loaded: {model.parameter_count():,} params, cap_feat_dim={cap_feat_dim}")

    return model, vae, text_encoder, tokenizer, clip_model, clip_tokenizer


def setup_lora(model, config):
    lora_rank = config['Model']['lora_rank']
    lora_alpha = config['Model']['lora_alpha']
    lora_target_modules = config['Model']['lora_target_modules']
    lora_dropout = config['Model'].get('lora_dropout', 0.05)
    cache_refined = config['Model'].get('cache_refined_cap_feats', False)

    modules_to_save = None
    if cache_refined:
        exclude_modules = [
            "cap_embedder",
            "context_refiner",
        ]

        filtered_modules = []
        for module_pattern in lora_target_modules:
            should_exclude = False
            for exclude_pattern in exclude_modules:
                if exclude_pattern in module_pattern:
                    should_exclude = True
                    break
            if not should_exclude:
                filtered_modules.append(module_pattern)

        lora_target_modules = filtered_modules
        logger.info(f"Refined cache enabled, excluding: {exclude_modules}")

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=modules_to_save,
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


def print_memory_usage(stage_name, profiler_enabled=False):
    """打印显存使用情况"""
    if not profiler_enabled:
        return

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3

        logger.info(f"=" * 80)
        logger.info(f"GPU Memory at [{stage_name}]")
        logger.info(f"  Allocated:     {allocated:.2f} GB")
        logger.info(f"  Reserved:      {reserved:.2f} GB")
        logger.info(f"  Max Allocated: {max_allocated:.2f} GB")
        logger.info(f"=" * 80)


def get_tensors_summary(profiler_enabled=False):
    """获取所有 CUDA 张量的摘要"""
    if not profiler_enabled or not torch.cuda.is_available():
        return

    import gc
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                tensors.append((type(obj).__name__, obj.size(), obj.dtype, obj.element_size() * obj.nelement() / 1024**2))
        except:
            pass

    # 按大小排序
    tensors.sort(key=lambda x: x[3], reverse=True)

    logger.info(f"\n{'='*80}")
    logger.info(f"Top 20 Largest Tensors in GPU Memory:")
    logger.info(f"{'Type':<20} {'Shape':<30} {'Dtype':<15} {'Size (MB)':<10}")
    logger.info(f"{'-'*80}")

    for i, (ttype, shape, dtype, size_mb) in enumerate(tensors[:20]):
        logger.info(f"{ttype:<20} {str(shape):<30} {str(dtype):<15} {size_mb:>10.2f}")

    total_size = sum(t[3] for t in tensors)
    logger.info(f"{'-'*80}")
    logger.info(f"Total tensor size: {total_size:.2f} MB ({len(tensors)} tensors)")
    logger.info(f"{'='*80}\n")


def compute_loss(model, vae, text_encoder, tokenizer, clip_model, clip_tokenizer, transport, batch, device, gemma3_prompt=""):
    if batch.get("cached", False):
        latents = batch["latents"].to(device)
        cap_feats = batch["cap_feats"].to(device)
        cap_mask = batch["cap_mask"].to(device)
        clip_text_pooled = batch["clip_text_pooled"].to(device)
        batch_size = latents.shape[0]
        use_refined = batch.get("refined", False)
    else:
        if vae is None or text_encoder is None or clip_model is None:
            raise RuntimeError("Models required for non-cached data but they are None. Enable cache or disable it in config.")

        pixel_values = batch["pixel_values"].to(device)
        captions = batch["captions"]
        batch_size = pixel_values.shape[0]

        with torch.no_grad():
            gemma_texts = [gemma3_prompt + cap if gemma3_prompt else cap for cap in captions]
            gemma_inputs = tokenizer(
                gemma_texts, padding=True, pad_to_multiple_of=8,
                truncation=True, max_length=512, return_tensors="pt"
            ).to(device)
            gemma_outputs = text_encoder(**gemma_inputs, output_hidden_states=True)
            cap_feats = gemma_outputs.hidden_states[-2]
            cap_mask = gemma_inputs.attention_mask

            clip_inputs = clip_tokenizer(
                captions, padding=True, truncation=True,
                max_length=2048, return_tensors="pt"
            ).to(device)
            clip_text_pooled = clip_model.get_text_features(**clip_inputs)

            latents = vae.encode(pixel_values).latent_dist.sample()
            scaling_factor = getattr(vae.config, 'scaling_factor', 0.13025)
            latents = latents * scaling_factor

        use_refined = False

    model_kwargs = dict(cap_feats=cap_feats, cap_mask=cap_mask, clip_text_pooled=clip_text_pooled, use_refined_cap_feats=use_refined)

    loss_dict = transport.training_losses(model, latents, model_kwargs)
    return loss_dict["loss"].mean()


def save_checkpoint(accelerator, model, optimizer, scheduler, step, config):
    """保存训练检查点"""
    checkpoint_dir = os.path.join(config['Model']['output_dir'], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")

    checkpoint = {
        "step": step,
        "lora_state_dict": get_peft_model_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
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
    parser.add_argument("--profiler", action="store_true", help="Enable GPU memory profiling")
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

    use_cache = config['Model'].get('use_cache', True)
    mixed_precision = config['Model'].get('mixed_precision', 'no')
    cache_dtype = torch.bfloat16 if mixed_precision == 'bf16' else (torch.float16 if mixed_precision == 'fp16' else torch.float32)
    gemma3_prompt = config['Model'].get('gemma3_prompt', '')
    resolution = config['Model']['resolution']

    if use_cache:
        logger.info("Checking if cache files exist...")
        train_data_dir = config['Model']['train_data_dir']
        image_paths = []
        for root, _, files in os.walk(train_data_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    image_paths.append(os.path.join(root, file))

        cache_complete = True
        for image_path in image_paths:
            vae_cache = f"{image_path}.safetensors"
            text_cache = f"{os.path.splitext(image_path)[0]}.txt.safetensors"
            if not os.path.exists(vae_cache) or not os.path.exists(text_cache):
                cache_complete = False
                break

        if cache_complete:
            logger.info("Cache complete, skipping encoder loading")
            vae = text_encoder = tokenizer = clip_model = clip_tokenizer = None
        else:
            logger.info("Cache incomplete, loading encoders for generation...")
            vae, text_encoder, tokenizer, clip_model, clip_tokenizer = load_encoders_only(config)

            dataset = ImageCaptionDataset(
                train_data_dir=train_data_dir,
                resolution=resolution,
                enable_bucket=config['Model'].get('enable_bucket', True),
                use_cache=True,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                clip_model=clip_model,
                clip_tokenizer=clip_tokenizer,
                device=accelerator.device,
                dtype=cache_dtype,
                gemma3_prompt=gemma3_prompt,
                cache_refined_cap_feats=config['Model'].get('cache_refined_cap_feats', False),
                max_token_length=config['Model'].get('max_token_length', 512),
            )

            logger.info("Cache generated, freeing encoders from GPU")
            del vae, text_encoder, clip_model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            vae = text_encoder = tokenizer = clip_model = clip_tokenizer = None

        logger.info("Loading NextDiT model for training...")
        model = load_transformer_only(config)
        print_memory_usage("After loading transformer", args.profiler)

        dataset = ImageCaptionDataset(
            train_data_dir=train_data_dir,
            resolution=resolution,
            enable_bucket=config['Model'].get('enable_bucket', True),
            use_cache=True,
            vae=None,
            text_encoder=None,
            tokenizer=None,
            clip_model=None,
            clip_tokenizer=None,
            device=accelerator.device,
            dtype=cache_dtype,
            gemma3_prompt=gemma3_prompt,
            cache_refined_cap_feats=config['Model'].get('cache_refined_cap_feats', False),
            max_token_length=config['Model'].get('max_token_length', 512),
        )
    else:
        model, vae, text_encoder, tokenizer, clip_model, clip_tokenizer = load_model_and_tokenizer(config)

        dataset = ImageCaptionDataset(
            train_data_dir=config['Model']['train_data_dir'],
            resolution=resolution,
            enable_bucket=config['Model'].get('enable_bucket', True),
            use_cache=False,
            vae=None,
            text_encoder=None,
            tokenizer=None,
            clip_model=None,
            clip_tokenizer=None,
            device=accelerator.device,
            dtype=cache_dtype,
            gemma3_prompt=gemma3_prompt,
            cache_refined_cap_feats=False,
            max_token_length=config['Model'].get('max_token_length', 512),
        )

    # 创建 Rectified Flow transport
    seq_len = (resolution // 16) ** 2
    transport = create_transport(
        path_type="Linear",
        prediction="velocity",
        snr_type="lognorm",
        do_shift=True,
        seq_len=seq_len
    )
    logger.info(f"Rectified Flow transport created (resolution={resolution}, seq_len={seq_len})")

    if config['Model'].get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    print_memory_usage("Before LoRA", args.profiler)
    model = setup_lora(model, config)
    print_memory_usage("After LoRA", args.profiler)

    num_workers = config['Model'].get('dataloader_num_workers', 4)
    prefetch_factor = config['Model'].get('dataloader_prefetch_factor', 2)
    batch_size = config['Model']['train_batch_size']

    if config['Model'].get('enable_bucket', True):
        batch_sampler = BucketBatchSampler(dataset, batch_size=batch_size, shuffle=True, seed=42)
        train_dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
    else:
        train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )

    optimizer = setup_optimizer(model, config)
    scheduler, num_training_steps = setup_scheduler(optimizer, config, train_dataloader)

    print_memory_usage("Before accelerator.prepare", args.profiler)
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)
    print_memory_usage("After accelerator.prepare", args.profiler)

    # Do NOT prepare encoders - they should stay frozen and not be wrapped
    if not use_cache:
        vae = vae.to(accelerator.device)
        text_encoder = text_encoder.to(accelerator.device)
        clip_model = clip_model.to(accelerator.device)
        vae.eval()
        text_encoder.eval()
        clip_model.eval()
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        clip_model.requires_grad_(False)
        print_memory_usage("After loading encoders (no cache)", args.profiler)

    start_step = load_checkpoint(accelerator, model, optimizer, scheduler, config)

    if args.profiler:
        get_tensors_summary(args.profiler)

    logger.info("Training started")
    global_step = start_step
    session_start_step = start_step

    if accelerator.is_main_process:
        accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])

    start_time = datetime.now()
    max_grad_norm = config['Optimization'].get('gradient_clip_norm', 1.0)
    save_epochs_interval = config['Model'].get('save_epochs_interval', 0)

    steps_per_epoch = len(train_dataloader)
    start_epoch = start_step // steps_per_epoch
    steps_to_skip_in_first_epoch = start_step % steps_per_epoch

    if start_step > 0:
        logger.info(f"Resuming from epoch {start_epoch+1}, will skip {steps_to_skip_in_first_epoch} steps in first epoch")

    for epoch in range(start_epoch, config['Model']['num_epochs']):
        if config['Model'].get('enable_bucket', True) and hasattr(train_dataloader, 'batch_sampler'):
            if hasattr(train_dataloader.batch_sampler, 'set_epoch'):
                train_dataloader.batch_sampler.set_epoch(epoch)

        epoch_loss_sum = torch.tensor(0.0, device=accelerator.device)
        epoch_loss_count = 0
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{config['Model']['num_epochs']}",
            disable=not accelerator.is_main_process
        )
        for batch_idx, batch in enumerate(progress_bar):
            if epoch == start_epoch and batch_idx < steps_to_skip_in_first_epoch:
                continue

            global_step += 1

            if global_step == 1 and args.profiler:
                print_memory_usage("Before first forward pass", args.profiler)

            loss = compute_loss(model, vae, text_encoder, tokenizer, clip_model, clip_tokenizer, transport, batch, accelerator.device, gemma3_prompt)
            epoch_loss_sum += loss.detach()
            epoch_loss_count += 1

            if global_step == 1 and args.profiler:
                print_memory_usage("After first forward pass", args.profiler)

            accelerator.backward(loss)

            if global_step == 1 and args.profiler:
                print_memory_usage("After first backward pass", args.profiler)

            if max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if global_step == 1 and args.profiler:
                print_memory_usage("After first optimizer step", args.profiler)
                get_tensors_summary(args.profiler)
                logger.info("Profiling complete for first step. You can now Ctrl+C to stop.")
                import sys
                sys.exit(0)

            if accelerator.is_main_process:
                loss_value = loss.item()
                accelerator.log({"loss": loss_value, "learning_rate": scheduler.get_last_lr()[0]}, step=global_step)

                elapsed = datetime.now() - start_time
                steps_in_session = global_step - session_start_step
                steps_per_sec = steps_in_session / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
                progress_bar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                    'speed': f'{steps_per_sec:.2f} steps/s'
                })

                if global_step % 100 == 0 or global_step == 1:
                    logger.info(f"Epoch {epoch+1}/{config['Model']['num_epochs']}, Step {global_step}/{num_training_steps}, Loss {loss_value:.4f}, LR {scheduler.get_last_lr()[0]:.7f}, Speed {steps_per_sec:.2f} steps/s")

            if global_step % 1000 == 0:
                save_checkpoint(accelerator, model, optimizer, scheduler, global_step, config)

        avg_epoch_loss = (epoch_loss_sum / epoch_loss_count).item() if epoch_loss_count > 0 else 0.0
        logger.info(f"Epoch {epoch+1}/{config['Model']['num_epochs']} completed - Average Loss: {avg_epoch_loss:.4f}")

        if save_epochs_interval == 0 or (epoch + 1) % save_epochs_interval == 0:
            save_checkpoint(accelerator, model, optimizer, scheduler, global_step, config)
            logger.info(f"Checkpoint saved at epoch {epoch+1}")

    logger.info("Training complete, saving final model")
    save_lora_model(accelerator, model, config)

    if accelerator.is_main_process:
        accelerator.end_training()

    logger.info("Training finished")


if __name__ == "__main__":
    main()
# Newbie LoRA Trainer

A LoRA training tool specifically designed for the Newbie (NextDiT_CLIP 3B) model.

## Features

- Automatic detection and loading of Diffusers-format checkpoints
- Dual text encoders (Gemma3-4B-IT + Jina CLIP v2)
- Variable-resolution training (resolution bucketing)
- VRAM optimization: gradient checkpointing, mixed precision, 8-bit optimizers
- Speed optimization: FlashAttention-2 support
- Resumable training: automatic detection and loading of checkpoints
- Three LoRA injection strategies (for 16 GB / 24 GB / 40 GB+ VRAM)

---

## Directory Structure

```
NewbieLoraTrainer/
├── models/                          # NextDiT_CLIP model architecture definitions
├── transport/                       # Rectified Flow implementation
├── train_newbie_lora.py            # Core training script
├── merge_lora.py                   # LoRA weight merging utility
├── convert_newbie_to_diffusers.py  # Model format conversion utility
├── config_template.toml            # Configuration template
├── requirements.txt                # Dependency list
└── README.md                       # This document
```

---

## Quickstart

### 1. Environment Setup

```bash
# Create a virtual environment (conda recommended)
conda create -n newbie-lora python=3.10
conda activate newbie-lora

# Install PyTorch (CUDA 12.1)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt

# Install 8-bit optimizer (recommended)
pip install bitsandbytes>=0.42.0
```

### 2. Model Conversion

Before training, you must first convert the model to Diffusers format:

```bash
python convert_newbie_to_diffusers.py \
    --checkpoint /path/to/consolidated.00-of-01.pth \
    --gemma3 google/gemma-3-4b-it \
    --jina jinaai/jina-clip-v2 \
    --output ./newbie-diffusers \
    --dtype bf16
```

### 3. Prepare Training Data

```
data/train/
├── 10_character/        # Directory name: repeatCount_description
│   ├── img1.jpg
│   ├── img1.txt        # Corresponding text caption
│   └── ...
└── 5_style/
    ├── img3.jpg
    └── img3.txt
```

### 4. Configure Training Parameters

```bash
cp config_template.toml my_config.toml
# Edit my_config.toml to set the model path and data path
```

Key configuration:

```toml
base_model_path = "./newbie-diffusers"
train_data_dir = "./data/train"
resolution = 1024
lora_rank = 32
lora_target_modules = ["attention.qkv", "attention.out", "feed_forward.w2", "time_text_embed.1", "clip_text_pooled_proj.1"]
```

### 5. Start Training

```bash
python train_newbie_lora.py --config_file my_config.toml
```

### 6. Merge LoRA

```bash
python merge_lora.py \
    --base_model_path ./newbie-diffusers \
    --lora_path ./output/my_lora \
    --output_path ./output/merged_model
```

---

## Configuration Guide

### LoRA Injection Strategies

#### Strategy 1: Minimal Injection (16 GB VRAM)

```toml
lora_target_modules = ["attention.qkv", "attention.out", "time_text_embed.1"]
```

Recommended settings:
- resolution = 768
- lora_rank = 16
- num_epochs = 100
- learning_rate = 5e-5

#### Strategy 2: Balanced Injection (24 GB VRAM, default)

```toml
lora_target_modules = [
    "attention.qkv",
    "attention.out",
    "feed_forward.w2",
    "time_text_embed.1",
    "clip_text_pooled_proj.1",
]
```

Recommended settings:
- resolution = 1024
- lora_rank = 32
- num_epochs = 50
- learning_rate = 1e-4

#### Strategy 3: Full Injection (40 GB+ VRAM)

```toml
lora_target_modules = [
    "attention.qkv",
    "attention.out",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
    "time_text_embed.1",
    "clip_text_pooled_proj.1",
    "adaLN_modulation.1",
    "cap_embedder.1",
]
```

Recommended settings:
- resolution = 1024
- lora_rank = 64
- train_batch_size = 2-4
- num_epochs = 30-50
- learning_rate = 2e-4

### VRAM Configuration Reference

| GPU Model | VRAM | batch_size | lora_rank | Strategy | Resolution |
|----------|------|------------|-----------|----------|------------|
| RTX 4060 Ti | 16 GB | 1 | 16 | Strategy 1 | 768 |
| RTX 3090/4090 | 24 GB | 1 | 32 | Strategy 2 | 1024 |
| A100 (40 GB) | 40 GB | 2-4 | 64 | Strategy 3 | 1024 |
| A100 (80 GB) | 80 GB | 4-8 | 64 | Strategy 3 | 1536 |

### Dataset Size Reference

| Number of Images | Recommended Epochs | Learning Rate | lora_rank |
|------------------|--------------------|---------------|-----------|
| < 100 | 100-200 | 5e-5 | 16 |
| 100-1000 | 50-100 | 1e-4 | 32 |
| > 1000 | 20-50 | 2e-4 | 64 |

---

## Advanced Features

### Resuming from Checkpoints

Simply rerun the training command; the script will automatically load the latest checkpoint:

```bash
python train_newbie_lora.py --config_file my_config.toml
```

### Tuning Hyperparameters

**Learning rate scheduler**:

```toml
lr_scheduler = "cosine"  # cosine/linear/constant
lr_warmup_steps = 100
```

**Optimizer**:

```toml
optimizer_type = "AdamW8bit"  # AdamW8bit/AdamW
gradient_clip_norm = 1.0
```

**Mixed precision**:

```toml
mixed_precision = "bf16"  # bf16/fp16/no
```

**LoRA parameters**:

```toml
lora_rank = 32
lora_alpha = 32
lora_dropout = 0.05
```

---

## FAQ

### Q: Out of memory (OOM) on the GPU?

Try the following steps in order:
1. `train_batch_size = 1`
2. `gradient_checkpointing = true`
3. `optimizer_type = "AdamW8bit"`
4. Use Strategy 1 (Minimal Injection)
5. Set `resolution = 768` or `512`
6. Set `lora_rank = 16`

### Q: Training is slow?

Possible optimizations:
1. Set `use_flash_attention_2 = true`
2. Set `mixed_precision = "bf16"`
3. Increase `train_batch_size` (if VRAM allows)
4. Check GPU utilization with `nvidia-smi`

### Q: Loss is not decreasing?

Check the following:
1. Learning rate (recommended range: 5e-5 ~ 2e-4)
2. Data quality (whether images and captions match)
3. Number of training epochs (small datasets require more epochs)
4. Whether the model path is correct

### Q: Multi-GPU training?

```bash
# Automatically detect all GPUs
python train_newbie_lora.py --config_file my_config.toml

# Or use Accelerate
accelerate launch --multi_gpu train_newbie_lora.py --config_file my_config.toml
```

---

## Technical Details

### NextDiT_CLIP Architecture

**Dual text encoders**:
- Gemma3-4B-IT: primary text features (`cap_feats`), max_length = 512, using `hidden_states[-2]`
- Jina CLIP v2: CLIP text features (`clip_text_pooled`), max_length = 2048

**Key modules**:
- `attention.qkv/out`: attention projections
- `feed_forward.w1/w2/w3`: SwiGLU feed-forward network
- `time_text_embed`: fusion of timestep and CLIP features
- `clip_text_pooled_proj`: projection of pooled CLIP features
- `adaLN_modulation`: adaptive layer normalization

**Rectified Flow training**:
- Objective: predict the velocity field
- Loss: MSE on velocity prediction
- Time-step sampling: uniform distribution + shift

### Training Pipeline

```
1. Data loading → ImageCaptionDataset
2. Model preparation → NextDiT_CLIP + Gemma3 + Jina CLIP + VAE
3. LoRA injection → use PEFT to attach LoRA adapters to target modules
4. Training loop:
   - Text encoding: Gemma3 + Jina CLIP
   - Image encoding: VAE → latents (×0.13025)
   - Rectified Flow loss computation
   - Backpropagation + gradient clipping
   - Checkpoint saving
```

### VRAM Usage (1024×1024, batch_size=1, bf16)

| Component | VRAM |
|----------|------|
| NextDiT_CLIP 3B (bf16) | ~6 GB |
| Gemma3-4B-IT (fp32, eval) | ~8 GB |
| Jina CLIP (fp32, eval) | ~2 GB |
| VAE (fp32, eval) | ~1 GB |
| LoRA adapters | ~50-200 MB |
| Optimizer (8-bit) | ~1-2 GB |
| Activations (checkpointed) | ~2-3 GB |
| **Total** | **~20-23 GB** |

Optimization tips:
- Enable gradient checkpointing: −3 GB
- Use 8-bit optimizer: −1 GB
- Lower the resolution (1024 → 768): −2 GB
- Use a more minimal injection strategy: −1 GB

---

## Related Resources

- [Newbie model](https://huggingface.co/NewBie-AI/NewBie-image-v0.1-exp-model-repo)
- [Gemma3-4B-IT](https://huggingface.co/google/gemma-3-4b-it)
- [Jina CLIP v2](https://huggingface.co/jinaai/jina-clip-v2)
- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [Rectified Flow paper](https://arxiv.org/abs/2210.02747)

---

## License

Apache 2.0

# Newbie LoRA 训练器

专为 Newbie (NextDiT_CLIP 3B) 模型设计的 LoRA 训练工具。

## 特性

- 支持 Diffusers 格式自动检测和加载
- 双文本编码器（Gemma3-4B-IT + Jina CLIP v2）
- 可变分辨率训练（分辨率分桶）
- 显存优化：梯度检查点、混合精度、8-bit 优化器
- 速度优化：FlashAttention-2 支持
- 断点续训：自动检测和加载检查点
- 三种 LoRA 注入策略（16GB/24GB/40GB+ 显存）

---

## 文件结构

```
NewbieLoraTrainer/
├── models/                          # NextDiT_CLIP 模型架构定义
├── transport/                       # Rectified Flow 实现
├── train_newbie_lora.py            # 核心训练脚本
├── merge_lora.py                   # LoRA 权重合并工具
├── convert_newbie_to_diffusers.py  # 模型格式转换工具
├── config_template.toml            # 配置模板
├── requirements.txt                # 依赖列表
└── README.md                       # 本文档
```

---

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境（推荐 conda）
conda create -n newbie-lora python=3.10
conda activate newbie-lora

# 安装 PyTorch（CUDA 12.1）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install -r requirements.txt

# 安装 8-bit 优化器（推荐）
pip install bitsandbytes>=0.42.0
```

### 2. 模型转换

训练前必须先转换模型为 Diffusers 格式：

```bash
python convert_newbie_to_diffusers.py \
    --checkpoint /path/to/consolidated.00-of-01.pth \
    --gemma3 google/gemma-3-4b-it \
    --jina jinaai/jina-clip-v2 \
    --output ./newbie-diffusers \
    --dtype bf16
```

### 3. 准备训练数据

```
data/train/
├── 10_character/        # 目录名：重复次数_描述
│   ├── img1.jpg
│   ├── img1.txt        # 对应的文本描述
│   └── ...
└── 5_style/
    ├── img3.jpg
    └── img3.txt
```

### 4. 配置训练参数

```bash
cp config_template.toml my_config.toml
# 编辑 my_config.toml，设置模型路径和数据路径
```

关键配置：
```toml
base_model_path = "./newbie-diffusers"
train_data_dir = "./data/train"
resolution = 1024
lora_rank = 32
lora_target_modules = ["attention.qkv", "attention.out", "feed_forward.w2", "time_text_embed.1", "clip_text_pooled_proj.1"]
```

### 5. 开始训练

```bash
python train_newbie_lora.py --config_file my_config.toml
```

### 6. 合并 LoRA

```bash
python merge_lora.py \
    --base_model_path ./newbie-diffusers \
    --lora_path ./output/my_lora \
    --output_path ./output/merged_model
```

---

## 配置指南

### LoRA 注入策略

#### 策略1：精简注入（16GB 显存）

```toml
lora_target_modules = ["attention.qkv", "attention.out", "time_text_embed.1"]
```

推荐配置：
- resolution = 768
- lora_rank = 16
- num_epochs = 100
- learning_rate = 5e-5

#### 策略2：平衡注入（24GB 显存，默认）

```toml
lora_target_modules = [
    "attention.qkv",
    "attention.out",
    "feed_forward.w2",
    "time_text_embed.1",
    "clip_text_pooled_proj.1",
]
```

推荐配置：
- resolution = 1024
- lora_rank = 32
- num_epochs = 50
- learning_rate = 1e-4

#### 策略3：全量注入（40GB+ 显存）

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

推荐配置：
- resolution = 1024
- lora_rank = 64
- train_batch_size = 2-4
- num_epochs = 30-50
- learning_rate = 2e-4

### 显存配置参考

| GPU 型号 | 显存 | batch_size | lora_rank | 策略 | 分辨率 |
|---------|------|------------|-----------|------|--------|
| RTX 4060 Ti | 16GB | 1 | 16 | 策略1 | 768 |
| RTX 3090/4090 | 24GB | 1 | 32 | 策略2 | 1024 |
| A100 (40GB) | 40GB | 2-4 | 64 | 策略3 | 1024 |
| A100 (80GB) | 80GB | 4-8 | 64 | 策略3 | 1536 |

### 数据集规模参考

| 图像数量 | 推荐轮数 | 学习率 | lora_rank |
|---------|---------|--------|-----------|
| < 100 | 100-200 | 5e-5 | 16 |
| 100-1000 | 50-100 | 1e-4 | 32 |
| > 1000 | 20-50 | 2e-4 | 64 |

---

## 高级功能

### 断点续训

直接重新运行训练命令，脚本会自动加载最新检查点：

```bash
python train_newbie_lora.py --config_file my_config.toml
```

### 调整超参数

**学习率调度**：
```toml
lr_scheduler = "cosine"  # cosine/linear/constant
lr_warmup_steps = 100
```

**优化器**：
```toml
optimizer_type = "AdamW8bit"  # AdamW8bit/AdamW
gradient_clip_norm = 1.0
```

**混合精度**：
```toml
mixed_precision = "bf16"  # bf16/fp16/no
```

**LoRA 参数**：
```toml
lora_rank = 32
lora_alpha = 32
lora_dropout = 0.05
```

---

## 常见问题

### Q: 显存不足 (OOM)？

按顺序尝试：
1. `train_batch_size = 1`
2. `gradient_checkpointing = true`
3. `optimizer_type = "AdamW8bit"`
4. 使用策略1（精简注入）
5. `resolution = 768` 或 `512`
6. `lora_rank = 16`

### Q: 训练速度慢？

优化方法：
1. `use_flash_attention_2 = true`
2. `mixed_precision = "bf16"`
3. 增加 `train_batch_size`（如果显存足够）
4. 检查 GPU 使用：`nvidia-smi`

### Q: 损失不下降？

检查：
1. 学习率（建议范围：5e-5 ~ 2e-4）
2. 数据质量（图像和文本是否匹配）
3. 训练轮数（小数据集需要更多轮数）
4. 模型路径是否正确

### Q: 多 GPU 训练？

```bash
# 自动检测所有 GPU
python train_newbie_lora.py --config_file my_config.toml

# 或使用 accelerate
accelerate launch --multi_gpu train_newbie_lora.py --config_file my_config.toml
```

---

## 技术细节

### NextDiT_CLIP 架构

**双文本编码器**：
- Gemma3-4B-IT：主文本特征（cap_feats），max_length=512，使用 hidden_states[-2]
- Jina CLIP v2：CLIP 文本特征（clip_text_pooled），max_length=2048

**关键层结构**：
- `attention.qkv/out`: 注意力投影
- `feed_forward.w1/w2/w3`: SwiGLU 前馈网络
- `time_text_embed`: 时间和 CLIP 特征融合
- `clip_text_pooled_proj`: CLIP 池化特征投影
- `adaLN_modulation`: 自适应层归一化

**Rectified Flow 训练**：
- 目标：预测 velocity field
- 损失：MSE (velocity prediction)
- 时间步采样：uniform 分布 + shift

### 训练流程

```
1. 数据加载 → ImageCaptionDataset
2. 模型准备 → NextDiT_CLIP + Gemma3 + Jina CLIP + VAE
3. LoRA 注入 → PEFT 在目标模块添加 LoRA adapter
4. 训练循环：
   - 文本编码：Gemma3 + Jina CLIP
   - 图像编码：VAE → latents (×0.13025)
   - Rectified Flow 损失计算
   - 反向传播 + 梯度裁剪
   - 检查点保存
```

### 显存占用（1024×1024, batch_size=1, bf16）

| 组件 | 显存 |
|------|------|
| NextDiT_CLIP 3B (bf16) | ~6 GB |
| Gemma3-4B-IT (fp32, eval) | ~8 GB |
| Jina CLIP (fp32, eval) | ~2 GB |
| VAE (fp32, eval) | ~1 GB |
| LoRA adapters | ~50-200 MB |
| Optimizer (8bit) | ~1-2 GB |
| Activations (checkpoint) | ~2-3 GB |
| **总计** | **~20-23 GB** |

优化建议：
- 梯度检查点：-3 GB
- 8-bit optimizer：-1 GB
- 降低分辨率 (1024→768)：-2 GB
- 精简策略：-1 GB

---

## 相关资源

- [Newbie 模型](https://huggingface.co/NewBie-AI/NewBie-image-v0.1-exp-model-repo)
- [Gemma3-4B-IT](https://huggingface.co/google/gemma-3-4b-it)
- [Jina CLIP v2](https://huggingface.co/jinaai/jina-clip-v2)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [Rectified Flow 论文](https://arxiv.org/abs/2210.02747)

---

## 许可证

Apache 2.0

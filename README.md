# nanoGPT-文本生成（天龙八部/唐诗）
基于nanoGPT实现的中文文本生成项目，支持训练唐诗/天龙八部文本的小尺寸GPT模型，生成风格化中文文本。

## 一、项目介绍
- 基于nanoGPT框架，适配CUDA 12.6环境
- 支持自定义中文文本训练（已适配唐诗、天龙八部文本）
- 模型参数：29.94M（6层+6头+384维嵌入），轻量化易训练
- 输出：训练好的模型权重、文本生成能力

## 二、环境配置
### 1. 依赖环境
- Python ≥ 3.9
- CUDA 12.6
- Anaconda（推荐）

### 2. 环境搭建
```bash
# 1. 创建虚拟环境
conda create -n gpt python=3.9 -y
conda activate gpt

# 2. 安装适配CUDA 12.6的PyTorch
pip install torch==2.4.0+cu126 torchaudio==2.4.0+cu126 torchvision==0.19.0+cu126 -f https://download.pytorch.org/whl/cu126/torch_stable.html

# 3. 安装其他依赖
pip install huggingface-hub==0.17.3 tokenizers==0.14.1 transformers==4.35.0 datasets==4.0.0 numpy==1.24.3 tiktoken==0.12.0 wandb==0.25.1


nanoGPT/
├─ data/
│  ├─ poemtext/          # 唐诗数据集
│  │  ├─ prepare.py      # 数据预处理脚本
│  │  ├─ tang_poet.txt   # 唐诗原始文本
│  │  └─ [生成的train.bin/val.bin]
│  └─ tianlong/          # 天龙八部数据集
│     ├─ prepare.py      # 数据预处理脚本
│     ├─ tianlong.txt    # 天龙八部原始文本
│     └─ [生成的train.bin/val.bin]
├─ config/
│  ├─ train_poemtext_char.py  # 唐诗训练配置
│  └─ train_tianlong_char.py  # 天龙八部训练配置
├─ train.py              # 主训练脚本
└─ README.md             # 项目说明


# 激活环境后，进入nanoGPT根目录
conda activate gpt
cd D:\大模型应用开发\案例三\nanoGPT

# 生成唐诗训练数据
python data/poemtext/prepare.py

# 生成天龙八部训练数据
python data/tianlong/prepare.py


# 加载唐诗配置文件启动训练
python train.py config/train_poemtext_char.py


# 加载天龙八部配置文件启动训练
python train.py config/train_tianlong_char.py



















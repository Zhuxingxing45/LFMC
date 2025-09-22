# 🧠 LFMC: Enhancing LLM Logical Reasoning through Mistake Correction

## 📖 项目简介

在近年来，**大语言模型（LLMs）** 由于其强大的文本生成能力而被广泛应用。但它们在逻辑推理中仍存在不足，尤其是**缺乏像人类一样通过错误反思来提升推理能力**。

本项目提出 **LFMC (Logic Fine-tuning with Mistake Correction)** 方法：

* 使用 GPT-4 自动修正包含逻辑错误的推理路径
* 构建 **LOCD (Logical Error Correction Dataset)** 数据集
* 基于 **LLaMA3-8B**，通过 **QLoRA** 高效微调提升逻辑推理能力

实验表明，使用 LOCD 微调后的模型在四个逻辑推理任务上均超越了基线模型，证明了 LLM 能够通过错误修正学习更稳健的逻辑推理。

---

## 🚀 功能特性

* 🔍 **逻辑错误修正**：利用 GPT-4 生成正确的推理路径
* 📊 **LOCD 数据集构建**：原始逻辑问题 + GPT-4 修正输出
* ⚡ **高效微调**：通过 QLoRA 对 LLaMA3-8B 进行参数高效微调
* 🧪 **实验验证**：在四个逻辑推理任务上显著提升性能

---

## 📦 安装与运行

### 环境要求

* Python >= 3.10
* CUDA >= 12.1 (建议使用 GPU 环境)
* 主要依赖：

  * [Transformers](https://github.com/huggingface/transformers)
  * [PEFT](https://github.com/huggingface/peft)
  * [Datasets](https://github.com/huggingface/datasets)
  * [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)

### 安装步骤

```bash
# 克隆仓库
git clone git@github.com:Zhuxingxing45/LFMC.git
cd LFMC

# 创建虚拟环境 (推荐)
conda create -n yourenv python=3.10
conda activate yourenv
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

---

## 📂 项目结构

```
LFMC/
├── data/                # 数据集（原始逻辑问题 & 修正后的推理路径）
├── root/              # 模型权重与配置
├── config/                # QLoRA 微调脚本
├── results/             # 实验结果
└── README.md            # 项目说明
```

---





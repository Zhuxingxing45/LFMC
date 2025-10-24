# 🧠 LFMC: Enhancing LLMs' Logical Reasoning through Mistake Correction

## 📖 项目简介

在近年来，**大语言模型（LLMs）** 由于其强大的文本生成能力而被广泛应用。但它们在逻辑推理中仍存在不足，尤其是**缺乏像人类一样通过错误反思来提升推理能力**。

本项目提出 **LFMC (Logic Fine-tuning with Mistake Correction)** 方法：

* 使用 GPT-4 自动修正包含逻辑错误的推理路径
* 构建 **LOCD (Logical Error Correction Dataset)** 数据集
* 使用LOCD，通过 **QLoRA** 高效微调提升多种大语言模型的逻辑推理能力

实验表明，使用 LOCD 微调后的模型在四个逻辑推理任务上均超越了基线模型，证明了 LLMs 能够通过错误修正学习更稳健的逻辑推理。

---

## 📂 Project Structure

```
LFMC/
├── data/                # Datasets (original logical questions & corrected reasoning paths)
├── root/                # Model weights and configurations
├── config/              # QLoRA fine-tuning scripts
├── results/             # Experimental results
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```


## 🚀 LOCD数据集构建

* 🔍 **逻辑错误修正**：利用 GPT-4 生成正确的推理路径
* 📊 **LOCD 数据集构建**：原始逻辑问题 + GPT-4 修正输出
* ⚡ **高效微调**：通过 QLoRA 对 LLaMA3-8B 进行参数高效微调
* 🧪 **实验验证**：在四个逻辑推理任务上显著提升性能

---

## 📦 Installation & Usage

### Environment

* GPU: NVIDIA GeForce RTX 4090
* Python >= 3.10
* CUDA >= 12.1 
* Dependencies: Listed in `requirements.txt`

### Installation

```bash
# Clone the repository:
git clone git@github.com:Zhuxingxing45/LFMC.git
cd LFMC

# 创建虚拟环境 (推荐)
conda create -n yourenv python=3.10
conda activate yourenv
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Usage
## Training
This section describes how to fine-tune a single model  with the training configuration.

### 1. Training a Single Model
All training configurations are located in the `configs` directory.  
For example, to train the LLaMA3-8B model with LOCD data:
```bash
# Train the model
xtuner train configs/llama/llama3_correction/llama3_8b_instruct_qlora_logic_correct_ez.py \
    --work-dir root/llama3-8b/llama3_logic_lfud/llama3_logic_original_pth
```

Convert checkpoint to HuggingFace adapter:
```bash
xtuner convert pth_to_hf <config_path> <checkpoint_path> <adapter_output_path>
```


Merge with base model (optional):

```bash
export MKL_SERVICE_FORCE_INTEL=1
xtuner convert merge <base_model_path> <adapter_path> <merged_model_path>
```

---

## **Evaluation**

Evaluate the fine-tuned model:

```bash
python logic_llm/qwen3/evaluate.py \
    --model_path <merged_model_path> \
    --output_path <generate_data_path> \
    --result_path <accuracy_json_path>
```

---

## **Data**

* LOCD dataset located in `./data/LOCD` (training/validation/test splits included)
* External datasets used: 
    * FOLIO: [https://github.com/Yale-LILY/FOLIO](https://github.com/Yale-LILY/FOLIO)
  * ReClor: [https://whyu.me/reclor/](https://whyu.me/reclor/)
  * LogiQA_v2: [https://github.com/csitfun/LogiQA2.0](https://github.com/csitfun/LogiQA2.0)
  * logiqa-zh: [https://doi.org/10.48550/arXiv.2007.08124](https://doi.org/10.48550/arXiv.2007.08124)
  * LogiCoT: [https://github.com/csitfun/LogiCoT](https://github.com/csitfun/LogiCoT)
  * LFUD: [https://github.com/YandaGo/LFUD](https://github.com/YandaGo/LFUD)
---









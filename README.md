# 🧠 LFMC: Enhancing LLMs' Logical Reasoning through Mistake Correction

## 📖 项目简介

In recent years, **Large Language Models (LLMs)** have been widely applied due to their powerful text generation capabilities. However, they still exhibit limitations in logical reasoning, particularly in their **inability to improve reasoning ability through human-like error reflection**.

This project proposes the **LFMC (Logic Fine-tuning with Mistake Correction)** method:

* Automatically corrects reasoning paths containing logical errors using GPT-4
* Constructs the **LOCD (Logical Error Correction Dataset)**
* Fine-tunes various LLMs efficiently with **QLoRA** using LOCD to enhance logical reasoning ability

Experimental results show that models fine-tuned with LOCD outperform baseline models across four logical reasoning tasks, demonstrating that LLMs can achieve more robust logical reasoning through mistake correction learning.


---

## 📂 Project Structure

```
LFMC/
├── data/                # Datasets (original logical questions & corrected reasoning paths)
├── root/                # Model weights and configurations
├── config/              # QLoRA fine-tuning scripts
├── logic_llm/           # generate and evaluate
│   ├── wrong_reasoning_colloction/    # Collect erroneous logical reasoning data.
│   ├── evaluate/         # evaluate
│   ├── results/          # Save the test results.
├── gpt4_correct/        # Using GPT-4 to correct data with faulty logical reasoning
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

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

### GPT-4 correcting
```bash
# Train the model
python gpt4_correct/models/logic_correction_v2.py --data_path ../Wrong_Inference --dataset_name [dataset_name] --api_key [openai_api_key]
```


### Training
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







# Copyright (c) OpenMMLab. All rights reserved.
import torch
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from xtuner.dataset import ConcatDataset, process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import (alpaca_map_fn, alpaca_zh_map_fn,
                                    template_map_fn_factory)
from xtuner.engine.hooks import (DatasetInfoHook, EvaluateChatHook,
                                 VarlenAttnArgsToMessageHubHook)
from xtuner.engine.runner import TrainLoop
from xtuner.model import SupervisedFinetune
from xtuner.utils import PROMPT_TEMPLATE, SYSTEM_TEMPLATE

#######################################################################
#                          Training Instruction                           
#xtuner train configs/internlm/internlm2-7b/internlm2_7b_qlora_lfud.py --work-dir root/internlm2/7b/internlm2_logic_lfud/internlm2_logic_original_pth
#xtuner convert pth_to_hf /home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/internlm2/7b/internlm2_logic_lfud/internlm2_logic_original_pth/internlm2_7b_qlora_lfud.py /home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/internlm2/7b/internlm2_logic_lfud/internlm2_logic_original_pth/iter_7182.pth /home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/internlm2/7b/internlm2_logic_lfud/internlm2_logic_original_hf_adapter
#export MKL_SERVICE_FORCE_INTEL=1
#xtuner convert merge /home/23_zxx/workspace/huggingface/internlm2-chat-7b /home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/internlm2/7b/internlm2_logic_lfud/internlm2_logic_original_hf_adapter /home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/internlm2/7b/internlm2_logic_lfud/internlm2_logic_original_hf_merged
#rm -rf /home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/internlm2/7b/internlm2_logic_lfud/internlm2_logic_original_hf_merged
#streamlit run /tools/internstudio_web_demo.py /root/llama3_logic_correct_ez_v6/llama3_logic_original_hf_merged
#######################################################################

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = '/home/23_zxx/workspace/huggingface/internlm2-chat-7b'
use_varlen_attn = False

# Data
data_files = [
            '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/qwen_format/origin/LogiQA_fintuing_data_formatted_base.json',
              '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/qwen_format/origin/Reclor_fintuing_data_formatted_base.json',
              '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/qwen_format/origin/FOLIO_fintuing_data_formatted_base.json',
              '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/qwen_format/origin/logiqa-zh_fintuing_data_formatted_base.json',

              '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/qwen_format/lfud/datasets.json',
              ]

prompt_template = PROMPT_TEMPLATE.default
max_length = 2048
pack_to_max_length = True

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 16
dataloader_num_workers = 0
max_epochs = 3
optim_type = AdamW
lr = 1e-4
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 500
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 500
SYSTEM = SYSTEM_TEMPLATE.alpaca
evaluation_inputs = [
    "Given the following premises:\nLawton Park is a neighborhood in Seattle. \nAll citizens of Lawton Park use the zip code 98199. \nTom is a citizen of Lawton Park.\nDaniel uses the zip code 98199. \nFor the following hypothesis:Tom doesn't use the zip code 98199.\n Which of the following options is correct? A)True, B)False, C)Uncertain \nPlease provide the correct option.",
    "Given the following premises:\nScreenwriter moviegoers are those who don't mind being spoiled by spoilers and even inquire about plot introductions and review all kinds of movies in advance. This kind of moviegoers pursue the feeling of controlling the development of the plot and don't like surprises.\nFor the following hypothesis:Xiao Li belongs to the screenwriter moviegoers according to the above definition, because he is fond of suspense movies, enjoys brain-burning plots, and assumes the role of a detective when watching movies. \n Which of the following options is correct? A)entailment, B)not-entailment \nPlease provide the correct option.",
    "Given the following context:\nAny sale item that is purchased can be returned for store credit but not for a refund of the purchase price. Every home appliance and every piece of gardening equipment is on sale along with selected construction tools.\nFor the following question:If the statements above are true, which one of the following must also be true?\n Which of the following options is correct? A)Any item that is not on sale cannot be returned for store credit., B)Some construction tools are not returnable for store credit., C)No piece of gardening equipment is returnable for a refund., D)None of the things that are returnable for a refund are construction tools.\nPlease provide the correct option.",
    "给定以下背景信息：\n有些广东人不爱吃辣椒. 因此,有些南方人不爱吃辣椒.\n对于以下问题：以下哪项能保证上述论证的成立？.\n  A)有些广东人爱吃辣椒.  B)爱吃辣椒的有些是南方人. C)所有的广东人都是南方人. D)有些广东人不爱吃辣椒也不爱吃甜食.\n请提供正确的选项。",
    "Given the following premises:\nRosa was born in Santiago. \nSantiago is the capital and largest city of Chile.\nRosa is the daughter of a Catalan building contractor, Jose.\nJose has a Chilean wife, Carmen.\nCarmen and Jose are Rosa's parents.\nPeople from Catalan are not from Chile.\nA building contractor is responsible for the day-to-day oversight of a construction site. \nFor the following hypothesis:Rosa is the daughter of someone who is responsible for the oversight of traffic.\nWhich of the following options is correct? A)True, B)False, C)Uncertain \nPlease provide the correct option and the reasoning process to verify this conclusion.\nThe original reasoning process is as follows:\n B. False\n\nReasoning process:\n\n1. Jose is a building contractor, and a building contractor is responsible for the day-to-day oversight of a construction site, not traffic.\n2. Therefore, Rosa is not the daughter of someone who is responsible for the oversight of traffic.\n\nSo, the correct answer is B) False.\nHowever, the correct option isA.Please identify and explain the mistakes in the original reasoning process, then correct these mistakes and provide the corrected final answer.",
    "直销是指直销企业招募直销员,由直销员在固定营业场所之外直接向最终消费者推销产品的经营方式.\n\n以下属于直销的是（）.\n\nA.某奶制品生产厂家甄选业务员后,在市内设立了一百个销售点以统一的价格销售奶制品\n\nB.某书店采取网上销售方式,顾客下订单后,由快递员将产品送至指定场所并收取费用\n\nC.某化妆品牌招聘的一些业务员在道路旁设摊点发放产品说明并以较低价格向顾客推销\n\nD.开学时,新生小贺到批发市场购买了五盏台灯,自用一盏,其他四盏卖给同学",
    "\nPassage: If you know a lot about history, it will be easy for you to impress people who are intellectuals. But unfortunately, you will not know much about history if you have not, for example, read a large number of history books. Ttherefore, if you are not well versed in history due to a lack of reading, it will not be easy for you to impress people who are intellectuals.\nQuestion: The argument's reasoning is flawed because the argument overlooks the possibility that\nA. it is more important to impress people who are not intellectuals than people who are intellectuals\nB. many intellectuals are not widely read in history\nC. there are other easy ways to impress intellectuals that do not involve knowing history\nD. there are people who learn about history who do not impress intellectuals\nAnswer and reasoning step by step:"
]


#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')

model = dict(
    type=SupervisedFinetune,
    use_varlen_attn=use_varlen_attn,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')),
    lora=dict(
        type=LoraConfig,
        r=64,
        lora_alpha=16,
        lora_dropout=0.2,
        bias='none',
        task_type='CAUSAL_LM'))

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path="json", data_files=data_files),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=alpaca_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn)


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        system=SYSTEM,
        prompt_template=prompt_template)
]

if use_varlen_attn:
    custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)

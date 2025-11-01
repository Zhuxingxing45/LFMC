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

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import alpaca_map_fn, template_map_fn_factory
from xtuner.engine.hooks import (DatasetInfoHook, EvaluateChatHook,
                                 VarlenAttnArgsToMessageHubHook)
from xtuner.engine.runner import TrainLoop
from xtuner.model import SupervisedFinetune
from xtuner.parallel.sequence import SequenceParallelSampler
from xtuner.utils import PROMPT_TEMPLATE, SYSTEM_TEMPLATE
# import os

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = '/path/to/Meta-Llama-3-8B-Instruct'
use_varlen_attn = False


data_files = [
    '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/LogiQA_v2/LogiQA_fintuing_data_formatted_base.json',
    '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/Reclor/Reclor_fintuing_data_formatted_base.json',
    '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/FOLIO/FOLIO_fintuing_data_formatted_base.json',
    '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/logiqa-zh/logiqa-zh_fintuing_data_formatted_base.json',
    '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/LogiCoT/mrc_formatted.json',
    '/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/LogiCoT/mrc_zh_formatted.json'
]
prompt_template = PROMPT_TEMPLATE.llama3_chat
max_length = 4096
pack_to_max_length = True

# parallel
sequence_parallel_size = 1

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 16
accumulative_counts *= sequence_parallel_size
dataloader_num_workers = 0
max_epochs = 3
optim_type = AdamW
lr = 2e-4
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
    "Context: sent1: leo is a kind of constellation sent2: to be found in means to be contained in sent3: move around means revolve sent4: if something is a part of something else then that something else contains that something sent5: being around something means surrounding that something sent6: a circle is a kind of shape sent7: revolving around something means orbiting that something sent8: a motion is a kind of event / action sent9: including means containing sent10: motion is when moves an object / something to a direction sent11: planets in the solar system orbit the sun sent12: approximately means about sent13: planets orbit stars sent14: how something appears is how that something looks sent15: throughout means over sent16: motion / movement means moving / to move sent17: a constellation contains stars sent18: years ( y ) are a metric unit used for measuring time generally used for values between 1 and 14000000000 sent19: similar means in common sent20: the solar system is a kind of system sent21: moving in a circle means moving in a circular motion sent22: a celestial body travelling around another celestial body means that celestial body completes a cycle around that other celestial body sent23: rotation is the circular movement of an object around a center / axis sent24: an orbit is a kind of regular path sent25: the earth revolving around the sun causes stars to appear in different areas in the sky at different times of year\nQuestion: Stars are organized into patterns called constellations. One constellation is named Leo. Which statement best explains why Leo appears in different areas of the sky throughout the year?",
    "If people perform in school talent shows often, then they attend and are very engaged with school events.",
    "\nPassage: Quality control investigator: Upon testing samples of products from our supplier that were sent by our field inspectors from various manufacturing locations, our laboratory discovered that over 20 percent of the samples were defective. Since our supplier is contractually required to limit the rate of defects among items it manufactures for us to below 5 percent, it has violated its contract with us.\nQuestion: The reasoning in the quality control investigator's argument is flawed in that the argument\nA. presumes, without providing justification, that the field inspectors were just as likely to choose a defective item for testing as they were to choose a nondefective item\nB. presumes, without providing justification, that the field inspectors made an equal number of visits to each of the various manufacturing sites of the supplier\nC. overlooks the possibility that the field inspectors tend to choose items for testing that they suspect are defective\nD. bases its conclusion on too small a sample of items tested by the laboratory\nAnswer and reasoning step by step:",
    "激励约束,即组织根据期望目标,人的行为规律,通过各种方式,去激发人的动力,使人有一股内在的动力和要求,迸发出积极性,主动性和创造性,同时规范人的行为,朝着组织所期望的目标前进的过程.\n\n根据上述定义,下列各项中属于激励约束的是（）.\n\nA.小王上班经常迟到,最近公司作出新规定,迟到者将被扣发整月奖金,小王不得不每天早起半个小时,果然一个月内再没有迟到过\n\nB.某学校组织青年教师讲课比赛,获胜者将被授予青年讲课标兵称号,尽管学校规定自愿报名,但青年教师仍然纷纷报名参加\n\nC.某公司鼓励员工自我增值,并向员工提供外出培训的机会,员工们响应的热情竟使得一时间公司人手有些紧张\n\nD.某公司新上任的李经理业务能力强,而且待人亲和有礼,很快获得了员工们的一致认可和好评",
    "If Thomas were a doctor, then it is overcast. If it snows, then Thomas is a doctor. If it is overcast, then William is an electrician. If William were an electrician, then it is overcast",
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
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'))

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
alpaca_en = dict(
    type=process_hf_dataset,
    # dataset=dict(type=load_dataset, path=alpaca_en_path),
    dataset=dict(type=load_dataset, path='json',data_files=data_files),
  
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=None,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn)

sampler = SequenceParallelSampler \
    if sequence_parallel_size > 1 else DefaultSampler
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=alpaca_en,
    sampler=dict(type=sampler, shuffle=True),
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

# # Copyright (c) OpenMMLab. All rights reserved.
# import torch
# from datasets import load_dataset
# from mmengine.dataset import DefaultSampler
# from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
#                             LoggerHook, ParamSchedulerHook)
# from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
# from peft import LoraConfig
# from torch.optim import AdamW
# from transformers import (AutoModelForCausalLM, AutoTokenizer,
#                           BitsAndBytesConfig)

# from xtuner.dataset import process_hf_dataset
# from xtuner.dataset.collate_fns import default_collate_fn
# from xtuner.dataset.map_fns import alpaca_map_fn, template_map_fn_factory
# from xtuner.engine.hooks import (DatasetInfoHook, EvaluateChatHook,
#                                  VarlenAttnArgsToMessageHubHook)
# from xtuner.engine.runner import TrainLoop
# from xtuner.model import SupervisedFinetune
# from xtuner.parallel.sequence import SequenceParallelSampler
# from xtuner.utils import PROMPT_TEMPLATE, SYSTEM_TEMPLATE

# import os


# #######################################################################
# #                          Training Instruction                           
# #xtuner train configs/llama3_origin/llama3_8b_instruct_qlora_logicot.py --work-dir /home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_logicot/llama3_logic_logicot_pth
# #xtuner convert pth_to_hf /home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_original_pth/llama3_8b_instruct_qlora_logic.py /home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_original_pth/iter_5868.pth /home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_original_hf_adapter
# #export MKL_SERVICE_FORCE_INTEL=1
# #xtuner convert merge /home/23_zxx/workspace/llama3-ft/Meta-Llama-3-8B-Instruct /home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_original_hf_adapter /home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_original_hf_merged
# #streamlit run /home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/tools/internstudio_web_demo.py /home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/root/llama3_logic_original_hf_merged

# #######################################################################



# #######################################################################
# #                          PART 1  Settings                           #
# #######################################################################
# # Model
# pretrained_model_name_or_path = '/home/23_zxx/workspace/llama3-ft/Meta-Llama-3-8B-Instruct'
# use_varlen_attn = False

# # Data
# # folder_path = '/home/23_zxx/project/LLM_Fintue/Llama3-Tutorial/data/LogiCoT'
# # files_name = os.listdir(folder_path)
# # data_files = [fname for fname in files_name if fname.endswith('_formatted.json')]
# # print(data_files)
# # data_files = data_files[:2]
# data_files = ['/home/23_zxx/workspace/llama3-ft/Llama3-Tutorial/data/LogiCoT/entailmentbank_formatted.json']

# prompt_template = PROMPT_TEMPLATE.llama3_chat
# max_length = 4096
# pack_to_max_length = True

# # parallel
# sequence_parallel_size = 1

# # Scheduler & Optimizer
# batch_size = 1  # per_device
# accumulative_counts = 16
# accumulative_counts *= sequence_parallel_size
# dataloader_num_workers = 0
# max_epochs = 3
# optim_type = AdamW
# lr = 2e-4
# betas = (0.9, 0.999)
# weight_decay = 0
# max_norm = 1  # grad clip
# warmup_ratio = 0.03

# # Save
# save_steps = 500
# save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

# # Evaluate the generation performance during the training
# evaluation_freq = 500
# SYSTEM = SYSTEM_TEMPLATE.alpaca
# evaluation_inputs = [
#     "Context: sent1: leo is a kind of constellation sent2: to be found in means to be contained in sent3: move around means revolve sent4: if something is a part of something else then that something else contains that something sent5: being around something means surrounding that something sent6: a circle is a kind of shape sent7: revolving around something means orbiting that something sent8: a motion is a kind of event / action sent9: including means containing sent10: motion is when moves an object / something to a direction sent11: planets in the solar system orbit the sun sent12: approximately means about sent13: planets orbit stars sent14: how something appears is how that something looks sent15: throughout means over sent16: motion / movement means moving / to move sent17: a constellation contains stars sent18: years ( y ) are a metric unit used for measuring time generally used for values between 1 and 14000000000 sent19: similar means in common sent20: the solar system is a kind of system sent21: moving in a circle means moving in a circular motion sent22: a celestial body travelling around another celestial body means that celestial body completes a cycle around that other celestial body sent23: rotation is the circular movement of an object around a center / axis sent24: an orbit is a kind of regular path sent25: the earth revolving around the sun causes stars to appear in different areas in the sky at different times of year\nQuestion: Stars are organized into patterns called constellations. One constellation is named Leo. Which statement best explains why Leo appears in different areas of the sky throughout the year?",
# ]

# #######################################################################
# #                      PART 2  Model & Tokenizer                      #
# #######################################################################
# tokenizer = dict(
#     type=AutoTokenizer.from_pretrained,
#     pretrained_model_name_or_path=pretrained_model_name_or_path,
#     trust_remote_code=True,
#     padding_side='right')

# model = dict(
#     type=SupervisedFinetune,
#     use_varlen_attn=use_varlen_attn,
#     llm=dict(
#         type=AutoModelForCausalLM.from_pretrained,
#         pretrained_model_name_or_path=pretrained_model_name_or_path,
#         trust_remote_code=True,
#         torch_dtype=torch.float16,
#         quantization_config=dict(
#             type=BitsAndBytesConfig,
#             load_in_4bit=True,
#             load_in_8bit=False,
#             llm_int8_threshold=6.0,
#             llm_int8_has_fp16_weight=False,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type='nf4')),
#     lora=dict(
#         type=LoraConfig,
#         r=16,
#         lora_alpha=16,
#         lora_dropout=0.1,
#         bias='none',
#         task_type='CAUSAL_LM'))

# #######################################################################
# #                      PART 3  Dataset & Dataloader                   #
# #######################################################################
# alpaca_en = dict(
#     type=process_hf_dataset,
#     # dataset=dict(type=load_dataset, path=alpaca_en_path),
#     dataset=dict(type=load_dataset, path='json',data_files=data_files),
  
#     tokenizer=tokenizer,
#     max_length=max_length,
#     dataset_map_fn=None,
#     template_map_fn=dict(
#         type=template_map_fn_factory, template=prompt_template),
#     remove_unused_columns=True,
#     shuffle_before_pack=True,
#     pack_to_max_length=pack_to_max_length,
#     use_varlen_attn=use_varlen_attn)

# sampler = SequenceParallelSampler \
#     if sequence_parallel_size > 1 else DefaultSampler
# train_dataloader = dict(
#     batch_size=batch_size,
#     num_workers=dataloader_num_workers,
#     dataset=alpaca_en,
#     sampler=dict(type=sampler, shuffle=True),
#     collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn))

# #######################################################################
# #                    PART 4  Scheduler & Optimizer                    #
# #######################################################################
# # optimizer
# optim_wrapper = dict(
#     type=AmpOptimWrapper,
#     optimizer=dict(
#         type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
#     clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
#     accumulative_counts=accumulative_counts,
#     loss_scale='dynamic',
#     dtype='float16')

# # learning policy
# # More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
# param_scheduler = [
#     dict(
#         type=LinearLR,
#         start_factor=1e-5,
#         by_epoch=True,
#         begin=0,
#         end=warmup_ratio * max_epochs,
#         convert_to_iter_based=True),
#     dict(
#         type=CosineAnnealingLR,
#         eta_min=0.0,
#         by_epoch=True,
#         begin=warmup_ratio * max_epochs,
#         end=max_epochs,
#         convert_to_iter_based=True)
# ]

# # train, val, test setting
# train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

# #######################################################################
# #                           PART 5  Runtime                           #
# #######################################################################
# # Log the dialogue periodically during the training process, optional
# custom_hooks = [
#     dict(type=DatasetInfoHook, tokenizer=tokenizer),
#     dict(
#         type=EvaluateChatHook,
#         tokenizer=tokenizer,
#         every_n_iters=evaluation_freq,
#         evaluation_inputs=evaluation_inputs,
#         system=SYSTEM,
#         prompt_template=prompt_template)
# ]

# if use_varlen_attn:
#     custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]

# # configure default hooks
# default_hooks = dict(
#     # record the time of every iteration.
#     timer=dict(type=IterTimerHook),
#     # print log every 10 iterations.
#     logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
#     # enable the parameter scheduler.
#     param_scheduler=dict(type=ParamSchedulerHook),
#     # save checkpoint per `save_steps`.
#     checkpoint=dict(
#         type=CheckpointHook,
#         by_epoch=False,
#         interval=save_steps,
#         max_keep_ckpts=save_total_limit),
#     # set sampler seed in distributed evrionment.
#     sampler_seed=dict(type=DistSamplerSeedHook),
# )

# # configure environment
# env_cfg = dict(
#     # whether to enable cudnn benchmark
#     cudnn_benchmark=False,
#     # set multi process parameters
#     mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
#     # set distributed parameters
#     dist_cfg=dict(backend='nccl'),
# )

# # set visualizer
# visualizer = None

# # set log level
# log_level = 'INFO'

# # load from which checkpoint
# load_from = None

# # whether to resume training from the loaded checkpoint
# resume = False

# # Defaults to use random seed and disable `deterministic`
# randomness = dict(seed=None, deterministic=False)

# # set log processor
# log_processor = dict(by_epoch=False)

# Frankstein

### 1. 项目概述
> 项目分三部分，其中MergeLM、mergekit 为开源项目，用于代码阅读、复现论文效果；scripts 中主要是下载模型和数据的脚本，整体项目目录如下

```shell

├── download_log
│   └── Platypus2-13B_20231225112710.log
├── mergekit
│   ├── examples
│   ├── mergekit
│   ├── notebook.ipynb
│   ├── output
│   ├── pyproject.toml
│   ├── README.md
│   └── run_mergekit.sh
├── MergeLM
│   ├── direct_inference_merged_llms_instruct_math_code.py
│   ├── figures
│   ├── inference_llms_instruct_math_code.py
│   ├── inference_plms_glue.py
│   ├── math_code_data
│   ├── merge_llms_instruct_math_code.py
│   ├── merge_plms_glue.py
│   ├── model_merging_methods
│   ├── models
│   ├── README.md
│   ├── save_gen_instruct_responses_results
│   ├── save_logs
│   ├── save_merge_llm_logs
│   ├── save_merge_models
│   ├── save_model_results
│   ├── test.ipynb
│   ├── train_plms_glue.py
│   └── utils
├── README.md
└── scripts
    ├── glue_download.py
    ├── huggingface_model_download.sh
    ├── rsync_model.sh
    └── run_mergekit.sh

```

### 2. 实验记录
> 使用MergeLM开源项目进行model merge，进行了如下三个实验，并产出了三个基于alpaca_eval 问题的答案；产出答案需要使用openai key 评测

```shell
# Scripts for Merging Models
# Example of merging WizardLM-13B-V1.2 and WizardMath-13B-V1.0 with Average Merging:
python merge_llms_instruct_math_code.py --merge_instruct --merge_math --merging_method_name average_merging --tensor_parallel_size 1


# Example of merging WizardLM-13B-V1.2 and WizardMath-13B-V1.0 with Task Arithmetic:
python merge_llms_instruct_math_code.py --merge_instruct --merge_math --merging_method_name task_arithmetic --scaling_coefficient 1.0 --tensor_parallel_size 1


#Example of merging WizardLM-13B-V1.2 and WizardMath-13B-V1.0 with Average Merging and DARE (drop rate 0.2):
python merge_llms_instruct_math_code.py --merge_instruct --merge_math --merging_method_name mask_merging --use_weight_rescale --weight_mask_rate 0.2 --mask_apply_method average_merging --tensor_parallel_size 1



# 产出的三个 alpaca_eval 问题的答案（按照上面顺序），需要使用 "alpaca_eval --model_outputs "$file" --annotators_config chatgpt_fn" 使用chatgpt 进行评测；评测前需要导入openaikey export OPENAI_API_KEY=""

/home/huyanwei/projects/Frankstein/MergeLM/save_gen_instruct_responses_results/instruct_math/alpaca_eval/average_merging.json


/home/huyanwei/projects/Frankstein/MergeLM/save_gen_instruct_responses_results/instruct_math/alpaca_eval/task_arithmetic_scaling_coefficient_1.0.json


/home/huyanwei/projects/Frankstein/MergeLM/save_gen_instruct_responses_results/instruct_math/alpaca_eval/mask_merging/average_merging/mask_0.2_0.2_rescale_True.json

```

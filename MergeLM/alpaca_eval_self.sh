#!/bin/bash
export OPENAI_API_KEY=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 

# file="./average_merging.json"
# alpaca_eval --model_outputs $file --annotators_config chatgpt_fn

# 文件数组
files=("./average_merging.json" "./mask_0.2_0.2_rescale_True.json" "./task_arithmetic_scaling_coefficient_1.0.json")

# 遍历文件数组
for file in "${files[@]}"
do
    echo "Processing $file..."
    alpaca_eval --model_outputs "$file" --annotators_config chatgpt_fn
done
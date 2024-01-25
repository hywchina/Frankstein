#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

# 检查至少提供了模型名称
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 model_name [local_directory]"
    exit 1
fi

# 读取传入的参数
repo_type=$1
model_name=$2
local_dir=$3  # 可选的本地目录参数

# 从模型名称中提取最后一个部分作为日志文件的一部分
log_model_name=$(basename "$model_name")


# 定义日志文件夹和生成日志文件名，格式：模型名称+时间戳
log_dir="download_log"
timestamp=$(date +"%Y%m%d%H%M%S")
log_file="${log_dir}/${log_model_name}_${timestamp}.log"

# 检查日志文件夹是否存在，如果不存在则创建
if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi


# 构建 huggingface-cli 命令
# command="huggingface-cli download --resume-download $model_name --token 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  --quiet"
# command="huggingface-cli download --resume-download $model_name --token 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"
command="huggingface-cli download --resume-download --repo-type $repo_type $model_name --token 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"

# 如果提供了本地目录，添加到命令中
if [ ! -z "$local_dir" ]; then
    command="$command --local-dir $local_dir --local-dir-use-symlinks True"
fi

# 使用 nohup 在后台运行命令并将输出重定向到日志文件
echo "$command"
nohup time bash -c "$command" > "$log_file" 2>&1 &

echo "Download of $model_name is running in background. Log file: $log_file"

# kill 下载进程 
# ps -ef |grep huggingface-cli|awk '{print $2}'|xargs kill
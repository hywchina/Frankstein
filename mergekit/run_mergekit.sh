#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"
# export HF_HUB_ENABLE_HF_TRANSFER=1

# 配置文件和输出路径
CONFIG_FILE="/home/huyanwei/projects/mergekit/examples/orcamini-platy-44layer.yml"
OUT_PATH="/home/huyanwei/projects/mergekit/merged/orcamini-platy-44layer"

# 可选参数
LORA_MERGE_CACHE="" # 例如 path/to/lora_merge_cache/
TRANSFORMERS_CACHE="/home/huyanwei/.cache/huggingface/hub" # 例如 path/to/transformers_cache/
CUDA="--cuda"  # defult: --no-cuda 或者使用  --cuda
LOW_CPU_MEMORY="--no-low-cpu-memory"  # default: --no-low-cpu-memory 或者使用 --low-cpu-memory
COPY_TOKENIZER="--copy-tokenizer"  # default: --copy-tokenizer 或者使用 --no-copy-tokenizer
ALLOW_CRIMES="--no-allow-crimes"  # default: no-allow-crimes 或者使用 --allow-crimes
OUT_SHARD_SIZE="5B" # default: 5B
VERBOSE="-v"  # 可以移除此行以关闭详细日志
TRUST_REMOTE_CODE="--no-trust-remote-code"  # default: no-trust-remote-code 或者使用 --trust-remote-code
CLONE_TENSORS="--no-clone-tensors"  # default: no-clone-tensors 或者使用 --clone-tensors
LAZY_UNPICKLE="--no-lazy-unpickle"  # default: no-lazy-unpickle 或者使用 --lazy-unpickle

# 运行 mergekit-yaml 命令
mergekit-yaml "$CONFIG_FILE" "$OUT_PATH" \
    --lora-merge-cache "$LORA_MERGE_CACHE" \
    --transformers-cache "$TRANSFORMERS_CACHE" \
    $CUDA \
    $LOW_CPU_MEMORY \
    $COPY_TOKENIZER \
    $ALLOW_CRIMES \
    --out-shard-size "$OUT_SHARD_SIZE" \
    $VERBOSE \
    $TRUST_REMOTE_CODE \
    $CLONE_TENSORS \
    $LAZY_UNPICKLE

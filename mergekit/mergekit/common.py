# Copyright (C) 2023 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import logging
import os
import os.path
from typing import List, Optional, Union

import huggingface_hub
import numpy as np
import peft
import torch
import transformers
from pydantic import BaseModel
from transformers import AutoConfig, PretrainedConfig

from mergekit.io import ShardedTensorIndex


class ModelReference(BaseModel, frozen=True):
    """A reference to a language model.

    Can be a hf hub path (username/repo), or local. Optionally includes a LoRA."""

    path: str
    lora_path: Optional[str] = None

    
    """这个方法的主要目的是将一个 LoRA 适配器合并到一个预训练模型中，并返回合并后模型的引用。
    它首先检查是否有 LoRA 路径指定，如果有，就在指定的缓存目录中创建输出路径，并加载原始模型
    和 LoRA 模型，将二者合并，然后将合并后的模型保存到输出路径。如果没有指定 LoRA 路径，则直
    接返回原始模型的引用。
    """
    def merged(
        self, cache_dir: Optional[str] = None, trust_remote_code: bool = False
    ) -> "ModelReference":
        """Merge the LoRA if applicable and return a reference to the result."""
        # 如果没有指定 LoRA 路径，则直接返回当前的模型引用。
        if not self.lora_path:
            return self

        # 如果没有指定缓存目录，抛出运行时错误。
        if not cache_dir:
            raise RuntimeError("Need to specify cache dir to merge adapters")

        # 构建输出路径，将缓存目录、模型路径和 LoRA 路径的基本名称组合起来。
        out_path = os.path.join(
            cache_dir,
            os.path.basename(self.path) + "_" + os.path.basename(self.lora_path),
        )

        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)
            logging.info(f"Loading {self.path} for merge...")

            # 从预训练路径加载模型，指定数据类型为 float16，优化内存使用。
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
            )

            # 加载 LoRA 模型，将其应用于已加载的模型。
            model = peft.PeftModel.from_pretrained(
                model, self.lora_path, is_trainable=False
            )
            logging.info(f"Merging {self.lora_path} into {self.path}")

            # 执行合并操作，并释放资源。
            model = model.merge_and_unload()
            model.save_pretrained(out_path, safe_serialization=True)
            del model

        return ModelReference(path=out_path)

    def config(self) -> PretrainedConfig:
        return AutoConfig.from_pretrained(self.path)

    def tensor_index(self, cache_dir: Optional[str] = None) -> ShardedTensorIndex:
        # 断言确保没有指定 LoRA 路径，因为这个方法不适用于有 LoRA 路径的情况。
        assert self.lora_path is None

        path = self.path
        if not os.path.exists(path):

            # 检查 Hugging Face 模型库中的路径是否包含带有 ".safetensors" 后缀的文件。
            has_safetensors = any(
                fn.lower().endswith(".safetensors")
                for fn in huggingface_hub.list_repo_files(path, repo_type="model")
            )
            patterns = ["tokenizer.model", "*.json"]
            if has_safetensors:
                patterns.append("*.safetensors")
            else:
                patterns.append("*.bin")
            print(f"patterns:{patterns}")

            # 从 Hugging Face 模型库下载模型文件到指定的缓存目录。
            path = huggingface_hub.snapshot_download(
                path, cache_dir=cache_dir, allow_patterns=patterns, resume_download=True
            )
            print(f"3path:{path}")
        
        # 从磁盘上指定的路径创建并返回一个 ShardedTensorIndex 对象。
        return ShardedTensorIndex.from_disk(path)

    @classmethod
    # 定义一个类方法 parse，用于解析一个字符串格式的模型引用。
    def parse(cls, value: str) -> "ModelReference":
        """Parse a ModelReference. Format: '<MODEL_PATH>(+<LORA_PATH>)?'"""

        chunks = value.split("+")
        # 如果只有一个部分，表示没有 LoRA 路径，直接创建并返回 ModelReference 实例。
        if len(chunks) == 1:
            return ModelReference(path=value)
        # 如果有两个部分，表示包含 LoRA 路径，创建并返回包含 LoRA 路径的 ModelReference 实例。
        elif len(chunks) == 2:
            return ModelReference(path=chunks[0], lora_path=chunks[1])
        raise ValueError(f"Can't parse {value}")

    def __str__(self) -> str:
        # 如果存在 LoRA 路径，返回包含 LoRA 路径的字符串表示。
        if self.lora_path:
            return f"{self.path}+{self.lora_path}"
        
        # 如果不存在 LoRA 路径，只返回模型路径的字符串表示。
        return self.path


def dtype_from_name(name: Optional[str]) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    elif name == "float16":
        return torch.float16
    elif name == "float32":
        return torch.float32
    raise RuntimeError(f'Unimplemented dtype "{name}"')


# 定义一个函数 rectify_embed_sizes，接收一个参数名和张量列表，用于调整嵌入层的尺寸。
def rectify_embed_sizes(param_name: str, tensors: List[torch.Tensor]):
    if "lm_head" in param_name or "embed_tokens" in param_name:
        # special case - if lm_head.weight or embed_tokens.weight have a size
        # mismatch, take the largest common submatrix of all of them
        # 检查参数名是否包含 "lm_head" 或 "embed_tokens"。
        # 这是特殊情况处理，针对语言模型头（lm_head）或嵌入标记（embed_tokens）的权重。
        if take_common_submatrix(tensors):
            # 调用 take_common_submatrix 函数来处理尺寸不匹配的情况。
            # 如果尺寸不一致，取所有张量中最大的公共子矩阵。
            logging.warning(
                f"Using common submatrix of size {tensors[0].shape} for {param_name}"
            )


def take_common_submatrix(tensors: List[torch.Tensor]) -> bool:
    """这个 take_common_submatrix 函数的目的是调整一系列 PyTorch 张量的尺寸，
    使它们具有相同的最小尺寸。这是通过取所有张量的最大共同子矩阵来实现的
    """
    min_size = [None, None]
    for t in tensors:
        for idx in range(2):
            # 对于每个张量的每个维度（假设张量是二维的）
            if min_size[idx] is None or t.shape[idx] < min_size[idx]:
                # 更新 min_size 列表，保留到目前为止遇到的最小尺寸。
                min_size[idx] = t.shape[idx]

    if not all(t.shape == torch.Size(min_size) for t in tensors):
        # 检查是否所有张量的尺寸都等于最小尺寸。
        for idx in range(len(tensors)):
            # 对于不符合最小尺寸的张量，进行裁剪。
            # 裁剪张量，使其尺寸与最小尺寸一致。
            tensors[idx] = tensors[idx][: min_size[0], : min_size[1]]
        return True
    return False


def gradient_weights(gradient: List[float], num_samples: int) -> List[float]:
    assert len(gradient) > 1, "Need at least two values to define gradient"

    samples_per_weight = num_samples // (len(gradient) - 1)

    res = []
    for y0, y1 in zip(gradient[:-1], gradient[1:]):
        res.extend(np.linspace(y0, y1, num=samples_per_weight))
    while len(res) < num_samples:
        res.append(gradient[-1])
    return res


def parse_kmb(value: Union[str, int]) -> int:
    if isinstance(value, int):
        return value
    elif value.isnumeric():
        return int(value)
    elif value[-1].lower() == "k":
        return int(value[:-1]) * 1000
    elif value[-1].lower() == "m":
        return int(value[:-1]) * 1000 * 1000
    elif value[-1].lower() == "b":
        return int(value[:-1]) * 1000 * 1000 * 1000
    else:
        raise ValueError(value)

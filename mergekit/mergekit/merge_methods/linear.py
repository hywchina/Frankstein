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

from typing import Dict

import torch

from mergekit.common import rectify_embed_sizes
from mergekit.config import ConfigReader
from mergekit.graph import TensorReference
from mergekit.merge_methods.base import MergeMethod


class LinearMerge(MergeMethod):
    # 定义 __call__ 方法，使得类的实例可以像函数那样被调用。
    def __call__(
        self,
        parameter_name: str,
        input_tensors: Dict[TensorReference, torch.Tensor],
        config: ConfigReader,
        **_kwargs,
    ) -> torch.Tensor:
        
        # 获取输入张量的键（即 TensorReference 对象）列表。
        keys = list(input_tensors.keys())

        # 从输入张量字典中提取所有张量。
        tensors = [input_tensors[key] for key in keys]

        # 对于每个张量，从配置中读取相应模型的权重。
        weights = [
            config.parameter("weight", model=key.model, required=True) for key in keys
        ]

        # 调用 rectify_embed_sizes 函数以确保所有张量的尺寸一致。
        rectify_embed_sizes(parameter_name, tensors)

        # 创建一个包含所有唯一张量形状的集合。
        unique_shapes = set(t.shape for t in tensors)
        # 如果存在不同的张量形状，抛出运行时错误。
        if len(unique_shapes) != 1:
            raise RuntimeError(
                f"Tensor size mismatch for {parameter_name}, sizes: {list(unique_shapes)}"
            )

        # 将所有张量堆叠成一个新的张量。
        tensors = torch.stack(tensors, dim=0)

        # 将权重列表转换为张量。
        weights = torch.tensor(weights, dtype=tensors.dtype, device=tensors.device)

        # 如果权重张量的维度少于张量堆叠的维度，则扩展权重张量的维度
        while len(weights.shape) < len(tensors.shape):
            weights.unsqueeze_(-1)

        # 计算加权张量的和。
        res = (weights * tensors).sum(dim=0)

        # 如果配置中指定了归一化，则对结果进行归一化处理。
        if config.parameter("normalize", default=True):
            res /= weights.sum(dim=0)

        return res

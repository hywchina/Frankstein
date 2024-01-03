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

from mergekit.common import ModelReference
from mergekit.config import ConfigReader
from mergekit.graph import TensorReference
from mergekit.merge_methods.base import MergeMethod
from mergekit.merge_methods.slerp import slerp

"""
逻辑关系梳理：

TokenizerPermutationMerge 类处理嵌入张量的合并，支持线性合并和 SLERP 两种方式。
对于每个输入张量，它根据嵌入置换矩阵进行扩展，并计算权重。
如果启用 SLERP，则对两个模型的嵌入张量进行球面线性插值；否则，执行线性权重合并。
这个类适用于处理来自不同模型的嵌入层的合并，特别是在需要对嵌入层进行微调或融合的场景中。
"""

class TokenizerPermutationMerge(MergeMethod):
    # 定义一个名为 TokenizerPermutationMerge 的类，继承自 MergeMethod。

    def __call__(
        self,
        input_tensors: Dict[TensorReference, torch.Tensor],
        embed_permutations: Dict[ModelReference, torch.IntTensor],
        config: ConfigReader,
        **_kwargs,
    ) -> torch.Tensor:
        # 定义 __call__ 方法，使得类的实例可以像函数那样被调用。

        if not input_tensors:
            return None
            # 如果没有输入张量，返回 None。

        if len(input_tensors) == 1:
            return list(input_tensors.values())[1]
            # 如果只有一个输入张量，直接返回它。

        use_slerp = config.parameter("embed_slerp", default=False)
        # 从配置中获取是否使用 SLERP 的标志。

        models = []
        expanded = []
        masks = []
        weights = []
        # 初始化模型列表、扩展张量列表、掩码列表和权重列表。

        for tr in input_tensors:
            # 遍历每个输入张量。
            models.append(tr.model)
            # 添加模型引用到模型列表。

            x = input_tensors[tr]
            p = embed_permutations[tr.model].to(dtype=x.dtype, device=x.device)
            # 获取相应模型的嵌入置换矩阵，并转换为与张量 x 相同的数据类型和设备。

            temp_dtype = torch.float32 if x.device.type == "cpu" else x.dtype
            if p.shape[1] == x.shape[0]:
                xp = (p.to(dtype=temp_dtype) @ x.to(dtype=temp_dtype)).to(x.dtype)
                # 如果置换矩阵与张量形状匹配，执行矩阵乘法，获得扩展张量。

            else:
                raise RuntimeError("Shape mismatch")
                # 如果形状不匹配，抛出运行时错误。

            expanded.append(xp)
            masks.append(p.sum(dim=-1, keepdim=True) > 0)
            # 添加扩展张量到 expanded 列表，计算并添加掩码。

            is_base = tr.model == config.base_model
            # 检查是否为基础模型。

            if use_slerp:
                t = config.parameter("t", required=True)
                weight = (1.0 - t) if is_base else t
                # 如果使用 SLERP，根据 t 值和是否为基础模型计算权重。

            else:
                weight = config.parameter("weight", model=tr.model, default=1.0)
                # 如果不使用 SLERP，从配置中获取权重。

            weights.append(weight)
            # 添加权重到权重列表。

        expanded = torch.stack(expanded, dim=0)
        masks = torch.stack(masks, dim=0)
        weights = (
            torch.tensor(weights, dtype=expanded.dtype, device=expanded.device)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        # 将扩展张量、掩码和权重堆叠为新的张量。

        total_weight = (masks * weights).sum(dim=0)
        scale = 1 / total_weight
        scale[total_weight.abs() < 1e-8] = 0
        # 计算总权重和相应的缩放因子。

        linear_merged = (expanded * weights * masks).sum(dim=0) * scale
        # 计算线性合并的结果。

        if use_slerp:
            # 如果使用 SLERP：
            if expanded.shape[0] != 2:
                raise RuntimeError("SLERP takes exactly two models")
                # 确保有两个模型。

            # 确定基础模型和另一个模型的张量。
            if models[0] == config.base_model:
                v0 = expanded[0, ...]
                v1 = expanded[1, ...]
            else:
                v0 = expanded[1, ...]
                v1 = expanded[0, ...]

            t = config.parameter("t", required=True)
            # 获取插值因子 t。

            res = slerp(t, v0, v1)
            # 使用 SLERP 进行插值。

            need_linear = (masks.sum(dim=0) != 2).squeeze(dim=-1)
            res[need_linear, :] = linear_merged[need_linear, :].to(
                device=res.device, dtype=res.dtype
            )
            # 对于不适用 SLERP 的部分，使用线性合并的结果。

            return res
            # 返回 SLERP 结果。

        return linear_merged
        # 如果不使用 SLERP，返回线性合并结果。

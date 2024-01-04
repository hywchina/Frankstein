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
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel
from typing_extensions import Literal

from mergekit.config import ConfigReader
from mergekit.graph import TensorReference
from mergekit.merge_methods.base import MergeMethod
from mergekit.sparsify import SparsificationMethod, sparsify


class ConsensusMethod(str, Enum):
    # 定义一个名为 ConsensusMethod 的枚举类，继承自 str 和 Enum

    count = "count"
    # 定义一个枚举成员，表示计数共识方法

    sum = "sum"
    # 定义另一个枚举成员，表示求和共识方法



class GeneralizedTaskArithmeticMerge(MergeMethod, BaseModel):
    # 定义一个名为 GeneralizedTaskArithmeticMerge 的类，继承自 MergeMethod 和 BaseModel

    consensus_method: Optional[ConsensusMethod]
    # 类的一个属性，表示共识方法

    sparsification_method: Optional[SparsificationMethod]
    # 类的另一个属性，表示稀疏化方法

    default_normalize: bool
    # 类的另一个属性，表示默认是否进行归一化


    def __call__(
        self,
        parameter_name: str,
        input_tensors: Dict[TensorReference, torch.Tensor],
        config: ConfigReader,
        **kwargs,
    ) -> torch.Tensor:
        # 实现 __call__ 方法，使得此类的实例可以像函数一样被调用

        # 收集任务向量
        tvs, base = get_task_vectors(
            parameter_name,
            config,
            input_tensors,
            required_parameters=["weight"],
            optional_parameters=["density"],
        )
        # 获取任务向量和基础张量

        if not tvs:
            return base
        # 如果没有任务向量，直接返回基础张量

        # 稀疏化
        if self.sparsification_method:
            for tv_info in tvs:
                tv_info["delta"] = sparsify(
                    tv_info["delta"],
                    density=tv_info.get("density", 1.0),
                    method=self.sparsification_method,
                )
        # 如果设置了稀疏化方法，对每个任务向量进行稀疏化处理

        deltas = torch.stack([tv["delta"] for tv in tvs], dim=0)
        weights = torch.tensor(
            [tv["weight"] for tv in tvs], dtype=deltas.dtype, device=deltas.device
        )
        # 将所有任务向量的变化量和权重堆叠成张量

        while len(deltas.shape) > len(weights.shape):
            weights.unsqueeze_(-1)
        # 确保权重张量的维度与变化量张量匹配

        weighted_deltas = deltas * weights
        # 计算加权变化量

        # 获取共识符号并混合变化量
        if self.consensus_method:
            mask_dtype = (
                torch.int8
                if config.parameter("int8_mask", default=False)
                else base.dtype
            )
            mask = get_mask(
                weighted_deltas, method=self.consensus_method, mask_dtype=mask_dtype
            )
            mixed_delta = (weighted_deltas * mask).sum(dim=0)
            divisor = (weights * mask).sum(dim=0)
            divisor[divisor == 0] = 1
        else:
            mixed_delta = weighted_deltas.sum(dim=0)
            divisor = weights.sum(dim=0)
            divisor[divisor.abs() < 1e-8] = 1
        # 根据共识方法计算混合变化量

        if config.parameter("normalize", default=self.default_normalize):
            mixed_delta /= divisor
        # 如果设置了归一化，则对混合变化量进行归一化

        return (base + mixed_delta).to(base.dtype)
        # 返回基础张量和混合变化量的和，转换为基础张量的数据类型



def get_task_vectors(
    parameter_name: str,
    config: ConfigReader,
    input_tensors: Dict[TensorReference, torch.Tensor],
    required_parameters: Optional[List[str]] = None,
    optional_parameters: Optional[List[str]] = None,
) -> Tuple[List[torch.Tensor], List[float], torch.Tensor]:
    
    """
    函数功能简述：

    get_task_vectors 函数用于计算不同模型参数与基础模型参数之间的差异（delta）。
    函数接收参数名称、配置读取器、输入张量字典以及必需和可选参数列表。
    它首先确定基础模型的张量，然后遍历其他模型的张量，计算与基础模型的差异。
    如果张量形状不匹配，函数会根据参数名称对张量进行裁剪或跳过处理。
    函数还会从配置中读取额外的必需和可选参数。
    最终，函数返回一个包含每个模型的差异信息和额外参数的列表，以及基础模型的张量。这些信息可用于后续的模型合并或调整操作。
    """

    # 定义函数 get_task_vectors，用于从输入张量中提取任务向量

    tensors = {tr.model: value for (tr, value) in input_tensors.items()}
    # 将输入张量字典转换为模型和张量值的映射

    keys = list(tensors.keys())
    # 获取模型键列表

    base = tensors[config.base_model]
    # 获取基础模型的张量

    res = []
    # 初始化结果列表

    for model in keys:
        # 遍历每个模型

        if model == config.base_model:
            continue
        # 跳过基础模型

        x = tensors[model].to(base.dtype)
        # 将当前模型的张量转换为与基础模型相同的数据类型

        if x.shape != base.shape:
            # 检查形状是否与基础模型匹配

            if "lm_head" in parameter_name or "embed_tokens" in parameter_name:
                x = x[: base.shape[0], : base.shape[1]]
                logging.warning(f"Using submatrix of {model}:{parameter_name}")
            else:
                logging.warning(
                    f"skipping {model}:{parameter_name} due to size mismatch"
                )
                continue
            # 如果形状不匹配，根据参数名称处理或跳过

        delta = x - base
        # 计算当前模型和基础模型之间的差异

        del x
        del tensors[model]
        # 删除已处理的张量以节省内存

        d = {}
        d["model"] = model
        d["delta"] = delta
        # 为当前模型创建一个字典，包含模型引用和差异张量

        for p in required_parameters:
            d[p] = config.parameter(p, model, required=True)
        # 获取并添加必需的参数

        for p in optional_parameters:
            d[p] = config.parameter(p, model, required=False)
        # 获取并添加可选的参数

        res.append(d)
        # 将当前模型的字典添加到结果列表中

    return res, base
    # 返回任务向量列表和基础张量



def get_mask(
    delta: torch.Tensor,
    method: Literal["sum", "count"] = "sum",
    mask_dtype: Optional[torch.dtype] = None,
):
    """
    返回一个掩码，用于确定哪些差异向量应该合并到最终模型中。

    使用 'sum' 方法是论文中描述的方法。使用 'count' 方法是一个简单的符号计数方法。

    函数功能简述：
    get_mask 函数根据输入的差异向量（delta）和指定的方法（sum 或 count）生成一个掩码。
    此掩码用于决定哪些差异向量应该被合并到最终模型中。
    如果使用 sum 方法，函数计算每个元素的符号与其绝对值的乘积之和，然后根据这个和的符号确定多数符号。
    如果使用 count 方法，函数仅根据符号的总和确定多数符号。
    函数最终返回一个布尔掩码，标识哪些元素的符号与多数符号一致。
    这种方法适用于在模型参数合并时根据多个模型参数的共识符号来决定合并策略。
    """
    # 定义函数 get_mask，用于根据给定的方法生成掩码

    if mask_dtype is None:
        mask_dtype = delta.dtype
    # 如果没有指定掩码数据类型，则使用 delta 的数据类型

    sign = delta.sign().to(mask_dtype)
    # 计算 delta 的符号并转换为指定的数据类型

    if method == "sum":
        sign_weight = (sign * delta.abs()).sum(dim=0)
        # 如果方法为 'sum'，则计算每个元素符号的加权和

        majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
        # 根据加权和的符号计算多数符号

        del sign_weight
        # 删除临时变量以节省内存

    elif method == "count":
        majority_sign = (sign.sum(dim=0) >= 0).to(mask_dtype) * 2 - 1
        # 如果方法为 'count'，则计算每个元素符号的总和，并根据其符号计算多数符号

    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')
        # 如果方法未实现，则抛出运行时错误

    return sign == majority_sign
    # 返回一个布尔掩码，指示 delta 的符号是否与多数符号一致

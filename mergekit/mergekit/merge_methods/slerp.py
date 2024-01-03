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

from typing import Dict, Union

import numpy as np
import torch

from mergekit.common import rectify_embed_sizes
from mergekit.config import ConfigReader
from mergekit.graph import TensorReference
from mergekit.merge_methods.base import MergeMethod


class SlerpMerge(MergeMethod):
    def __call__(
        self,
        parameter_name: str,
        input_tensors: Dict[TensorReference, torch.Tensor],
        config: ConfigReader,
        **kwargs,
    ) -> torch.Tensor:
        if len(input_tensors) == 1:
            return list(input_tensors.values())[0]
        elif len(input_tensors) != 2:
            raise RuntimeError("Slerp merge expects exactly two models")

        # 获取两个输入张量的键值对
        [a, b] = list(input_tensors.items())
        if a[0].model != config.base_model:
            # 确保第一个张量（a）来源于基础模型。
            [a, b] = [b, a]

        # 准备要合并的张量。
        prepped_tensors = [a[1], b[1]]

        # 调整张量的尺寸以确保它们具有相同的尺寸。
        rectify_embed_sizes(parameter_name, prepped_tensors)

        # 使用 slerp 函数进行球面线性插值，并确保结果张量与输入张量具有相同的数据类型和设备。
        return (
            slerp(
                config.parameter("t", required=True),
                prepped_tensors[0],
                prepped_tensors[1],
            )
            .to(prepped_tensors[0].dtype)
            .to(prepped_tensors[0].device)
        )


def lerp(
    t: float, v0: Union[np.ndarray, torch.Tensor], v1: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    
    # 定义一个名为 lerp 的函数，用于执行线性插值。
    # 参数 t 是插值因子，v0 和 v1 是插值的起始和结束值，可以是 NumPy 数组或 PyTorch 张量。
    # 返回插值结果：根据 t 的值，从 v0 向 v1 线性插值。
    return (1 - t) * v0 + t * v1

def slerp(
    t: Union[float, np.ndarray],
    v0: Union[np.ndarray, torch.Tensor],
    v1: Union[np.ndarray, torch.Tensor],
    DOT_THRESHOLD: float = 0.9995,
    eps: float = 1e-8,
):
    """
    球面线性插值

    参数：
        t (float/np.ndarray): 浮点值，在0.0和1.0之间
        v0 (np.ndarray): 起始向量
        v1 (np.ndarray): 结束向量
        DOT_THRESHOLD (float): 用于判断两个向量是否共线的阈值。不推荐修改此值。
    返回：
        v2 (np.ndarray): v0 和 v1 之间的插值向量
    """
    is_torch = False
    if not isinstance(v0, np.ndarray):
        is_torch = True
        v0 = v0.detach().cpu().float().numpy()
    if not isinstance(v1, np.ndarray):
        is_torch = True
        v1 = v1.detach().cpu().float().numpy()
    # 判断输入向量是否为 PyTorch 张量，并将它们转换为 NumPy 数组。

    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    # 复制向量以便稍后使用。

    v0 = normalize(v0, eps)
    v1 = normalize(v1, eps)
    # 归一化向量以获取方向和角度。

    dot = np.sum(v0 * v1)
    # 计算归一化向量的点积。

    if np.abs(dot) > DOT_THRESHOLD:
        res = lerp(t, v0_copy, v1_copy)
        return maybe_torch(res, is_torch)
    # 如果点积的绝对值接近1，表示向量几乎共线，此时使用线性插值。

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # 计算 v0 和 v1 之间的初始角度。

    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    # 计算时间步 t 时的角度。

    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    res = s0 * v0_copy + s1 * v1_copy
    # 完成 slerp 算法，计算插值结果。

    return maybe_torch(res, is_torch)
    # 返回插值结果，如果原始向量为 PyTorch 张量，则将结果转换回 PyTorch 张量。

def maybe_torch(v: np.ndarray, is_torch: bool):
    # 定义一个函数 maybe_torch，它接收一个 NumPy 数组和一个布尔值 is_torch。

    if is_torch:
        return torch.from_numpy(v)
        # 如果 is_torch 为 True，则将 NumPy 数组转换为 PyTorch 张量并返回。

    return v
    # 如果 is_torch 为 False，直接返回原始的 NumPy 数组。

def normalize(v: np.ndarray, eps: float):
    # 定义一个函数 normalize，它接收一个 NumPy 数组和一个浮点数 eps。

    norm_v = np.linalg.norm(v)
    # 计算 v 的 L2 范数（欧几里得范数）。

    if norm_v > eps:
        v = v / norm_v
        # 如果范数大于 eps，将 v 归一化（即 v 除以其范数）。

    return v
    # 返回归一化后的向量。
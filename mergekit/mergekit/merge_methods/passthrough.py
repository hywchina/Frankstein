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

from mergekit.config import ConfigReader
from mergekit.graph import TensorReference
from mergekit.merge_methods.base import MergeMethod


class PassthroughMerge(MergeMethod):
    # 定义 __call__ 方法，使得类的实例可以像函数那样被调用。
    def __call__(
        self,
        parameter_name: str,
        input_tensors: Dict[TensorReference, torch.Tensor],
        config: ConfigReader,
        **_kwargs,
    ) -> torch.Tensor:
        # 如果输入张量不是恰好一个，抛出运行时错误。
        if len(input_tensors) != 1:
            raise RuntimeError("Passthrough merge expects exactly one tensor")
        
        # 获取输入张量字典中的第一个（也是唯一的）张量。
        res = list(input_tensors.values())[0]
        scale = config.parameter("scale")
        if scale is not None:
            res *= scale

        return res

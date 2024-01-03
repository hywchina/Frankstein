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

from abc import ABC, abstractmethod
from typing import Dict, Sequence

import torch
from transformers import PretrainedConfig

from mergekit.common import ModelReference
from mergekit.config import ConfigReader, MergeConfiguration
from mergekit.graph import TensorReference

"""

MergeMethod 类是一个抽象基类（ABC），它定义了合并模型参数的基本结构和方法。
这个类提供了用于合并操作的基本框架，并可以被不同的具体合并策略所继承和实现。
以下是对这个类的逐行中文注释：
"""

class MergeMethod(ABC):
    # 定义一个名为 MergeMethod 的抽象基类。

    @abstractmethod
    def __call__(
        self,
        parameter_name: str,
        input_tensors: Dict[TensorReference, torch.Tensor],
        config: ConfigReader,
        **kwargs,
    ) -> torch.Tensor:
        # 定义一个抽象方法 __call__，这使得类的实例可以像函数那样被调用。
        # 这个方法需要在子类中被具体实现。
        ...

    def general_dependencies(self) -> Sequence[TensorReference]:
        """List any tensors necessary for *every* merge operation"""
        # 定义一个方法 general_dependencies，用于列出每次合并操作都需要的张量。
        return []
        # 默认返回一个空列表。

    def input_layer_dependencies(
        self, model: ModelReference, layer_idx: int
    ) -> Sequence[TensorReference]:
        """List any tensors necessary when input includes a specific layer"""
        # 定义一个方法 input_layer_dependencies，用于列出当输入包含特定层时所需的张量。
        return []
        # 默认返回一个空列表。

    def model_out_config(self, config: MergeConfiguration) -> PretrainedConfig:
        """Return a configuration for the resulting model."""
        # 定义一个方法 model_out_config，用于返回合并后模型的配置。
        if config.base_model:
            res = ModelReference.parse(config.base_model).config()
            # 如果配置中有基础模型，解析并获取该模型的配置。
        else:
            res = config.referenced_models()[0].config()
            # 否则，获取第一个参考模型的配置。

        if config.dtype:
            res.torch_dtype = config.dtype
            # 如果配置中指定了数据类型，设置结果模型的数据类型。
        return res
        # 返回配置结果。

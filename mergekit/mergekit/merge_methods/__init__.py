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

from mergekit.merge_methods.base import MergeMethod
# 导入 MergeMethod 基类

from mergekit.merge_methods.generalized_task_arithmetic import (
    ConsensusMethod,
    GeneralizedTaskArithmeticMerge,
    SparsificationMethod,
)
# 导入广义任务算术合并方法及其相关类

from mergekit.merge_methods.linear import LinearMerge
# 导入线性合并方法

from mergekit.merge_methods.passthrough import PassthroughMerge
# 导入透传合并方法

from mergekit.merge_methods.slerp import SlerpMerge
# 导入球面线性插值合并方法

from mergekit.merge_methods.tokenizer_permute import TokenizerPermutationMerge
# 导入分词器置换合并方法


def get(method: str) -> MergeMethod:
    # 定义一个函数 get，接受一个方法名称字符串，返回相应的合并方法对象

    if method == "linear":
        return LinearMerge()
    # 如果方法为 linear，则返回 LinearMerge 对象

    elif method == "slerp":
        return SlerpMerge()
    # 如果方法为 slerp，则返回 SlerpMerge 对象

    elif method == "passthrough":
        return PassthroughMerge()
    # 如果方法为 passthrough，则返回 PassthroughMerge 对象

    elif method == "task_arithmetic":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=None,
            sparsification_method=None,
            default_normalize=False,
        )
    # 如果方法为 task_arithmetic，则返回 GeneralizedTaskArithmeticMerge 对象，不使用共识和稀疏化方法

    elif method == "ties":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.magnitude,
            default_normalize=True,
        )
    # 如果方法为 ties，则返回 GeneralizedTaskArithmeticMerge 对象，使用求和共识和幅值稀疏化方法

    elif method == "dare_ties":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.rescaled_random,
            default_normalize=False,
        )
    # 如果方法为 dare_ties，则返回 GeneralizedTaskArithmeticMerge 对象，使用求和共识和重缩放随机稀疏化方法

    elif method == "dare_linear":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=None,
            sparsification_method=SparsificationMethod.rescaled_random,
            default_normalize=False,
        )
    # 如果方法为 dare_linear，则返回 GeneralizedTaskArithmeticMerge 对象，不使用共识但使用重缩放随机稀疏化方法

    raise RuntimeError(f"Unimplemented merge method {method}")
    # 如果方法未实现，则抛出运行时错误


__all__ = [
    "MergeMethod",
    "get",
    "LinearMerge",
    "SlerpMerge",
    "PassthroughMerge",
    "GeneralizedTaskArithmeticMerge",
    "TokenizerPermutationMerge",
]

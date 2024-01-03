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

"""
Computational graph execution for tensor operations

This module provides a mechanism for constructing and executing a computational
graph for operations on tensors. The tensors are computed lazily,
being loaded and operated upon as per the defined computation graph
and execution strategy.

The primary class, `Executor`, uses a `RuleSet` to build a computation graph,
organizes an execution order which minimizes tensor resource requirements, and
executes the operations, handling tensor loading and storage automatically.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import networkx
import torch
import tqdm
from pydantic import BaseModel
from typing_extensions import Protocol

from mergekit.common import ModelReference
from mergekit.io import LazyTensorLoader, TensorWriter


class TensorReference(BaseModel, frozen=True):
    """
    张量的引用，可选择性地与特定模型关联。

    属性:
    - model: 可选的语言模型引用。
    - key: 张量的字符串标识符。

    类功能简述：
    TensorReference 类用于创建对张量的引用，其中可以包含对特定语言模型的引用（如果有的话）以及张量的唯一标识符（key）。
    model 属性是对语言模型的可选引用，如果没有特定模型关联，则此属性可以为 None。
    key 属性是对张量的唯一字符串标识符。
    __str__ 方法提供了类的字符串表示，用于方便地打印和标识张量引用。如果对象有模型引用，它会包含在字符串表示中；否则，使用下划线作为命名空间的占位符。

    """

    model: Optional[ModelReference]
    # 类的一个属性，表示对语言模型的可选引用

    key: str
    # 类的另一个属性，表示张量的字符串标识符

    def __str__(self) -> str:
        # 定义一个方法，将对象转换为字符串表示

        if self.model is not None:
            namespace = str(self.model)
        else:
            namespace = "_"
        # 如果有模型引用，则使用模型的字符串表示作为命名空间，否则使用下划线

        return namespace + ":" + self.key
        # 返回命名空间和张量标识符的组合



class Operation(BaseModel, frozen=True):
    """
    定义计算图中的一个节点，代表对张量的操作。

    属性:
    - function: 要执行的操作的字符串标识符。
    - inputs: 此操作的张量输入列表。
    - kwargs: 操作的可选关键字参数。
    """

    function: str
    # 类的一个属性，表示要执行的操作的字符串标识符

    inputs: List[TensorReference]
    # 类的另一个属性，表示此操作的张量输入列表

    kwargs: Optional[Dict[str, Any]] = None
    # 类的另一个属性，表示操作的可选关键字参数，是一个字典，默认为 None



class ProceduralRule(ABC):
    """
    抽象基类，用于过程性规则。过程性规则定义了一种方法，用于动态生成能够产生给定
    `TensorReference` 的 `Operation` 实例。
    """

    @abstractmethod
    def can_generate_rule(self, component: TensorReference) -> bool:
        # 定义一个抽象方法，用于判断是否可以为给定的 TensorReference 生成规则
        ...

    @abstractmethod
    def generate_rule(self, component: TensorReference) -> Optional[Operation]:
        # 定义另一个抽象方法，用于为给定的 TensorReference 生成 Operation 实例
        ...

class LoadTensorRule(ProceduralRule):
    """用于从输入模型加载张量的规则。"""

    model: ModelReference
    # 类的一个属性，表示模型的引用

    tensor_paths: Dict[str, str]
    # 类的另一个属性，表示张量路径的字典，键是张量的标识符，值是张量的路径

    def __init__(
        self,
        model: ModelReference,
        tensor_paths: Dict[str, str],
        dtype: Optional[str],
    ):
        # 类的构造函数
        self.model = model
        self.tensor_paths = tensor_paths
        self.dtype = dtype
        # 初始化模型引用、张量路径和数据类型

    def can_generate_rule(self, component: TensorReference) -> bool:
        # 实现 can_generate_rule 方法，判断是否可以为给定的 TensorReference 生成规则
        return (
            isinstance(component, TensorReference)
            and component.model == self.model
            and component.key in self.tensor_paths
        )
        # 如果 component 是 TensorReference 类型，且其模型和键都在当前规则的范围内，则返回 True

    def generate_rule(self, component: TensorReference) -> Operation:
        # 实现 generate_rule 方法，为给定的 TensorReference 生成 Operation
        if not self.can_generate_rule(component):
            return None
        # 如果不符合生成规则的条件，返回 None

        return Operation(
            function="load_tensor",
            inputs=[],
            kwargs={
                "model": component.model,
                "key": component.key,
                "dtype": self.dtype,
            },
        )
        # 返回一个 Operation 实例，函数为 "load_tensor"，输入为空，关键字参数包括模型引用、键和数据类型



class RuleSet:
    """
    A mapping from TensorReference instances to specific Operations to produce them.

    Can contain both statically defined rules and procedural rules for dynamic
    operation generation.
    """

    static: Dict[TensorReference, Operation]
    procedural: List[ProceduralRule]

    def __init__(
        self,
        static: Optional[Dict[TensorReference, Operation]] = None,
        procedural: Optional[List[ProceduralRule]] = None,
    ):
        self.static = static or {}
        self.procedural = procedural or []

    def get(self, tensor: TensorReference) -> Optional[Operation]:
        """
        Retrieve an operation to produce the specified tensor.

        First checks if a static operation exists for the given tensor reference.
        If not, iterates over procedural rules to find a match.
        """
        if tensor in self.static:
            return self.static[tensor]

        for proc_rule in self.procedural:
            if proc_rule.can_generate_rule(tensor):
                operation = proc_rule.generate_rule(tensor)
                if operation:
                    return operation
        return None


class OperationProtocol(Protocol):
    """操作实现的协议。"""

    def __call__(
        self, tensors: Dict[TensorReference, torch.Tensor], **kwargs
    ) -> Optional[torch.Tensor]:
        ...
        # 定义操作的调用方式，接受张量字典和关键字参数，返回可能的张量


def _normalized_shard_name(path: str) -> int:
    name, _ext = os.path.splitext(os.path.basename(path))
    return name.lower().replace("pytorch_model", "model")
    # 从路径中提取文件名，转为小写，并替换特定子串，以规范化分片名称


class Executor:
    """
    主要计算管理器，负责以结构化和资源最小化的方式组织和执行张量操作。

    `Executor` 接收模型、目标张量引用、规则和操作定义，以创建和执行计算图。
    """

    rules: RuleSet
    loaders: Dict[ModelReference, LazyTensorLoader]
    targets: List[TensorReference]
    operations: Dict[str, OperationProtocol]
    low_cpu_memory: bool = False
    # 类的属性，包括规则集、加载器、目标张量、操作和是否使用低内存模式

    def __init__(
        self,
        models: List[ModelReference],
        targets: List[TensorReference],
        rules: RuleSet,
        operations: Optional[Dict[str, OperationProtocol]] = None,
        transformers_cache_dir: Optional[str] = None,
        lora_cache_dir: Optional[str] = None,
        dtype: Optional[str] = None,
        cuda: bool = False,
        low_cpu_memory: bool = False,
        trust_remote_code: bool = False,
        lazy_unpickle: bool = False,
    ):
       # 构造函数，初始化 Executor
        if lora_cache_dir is None and transformers_cache_dir is not None:
            lora_cache_dir = transformers_cache_dir
        # 如果没有指定 lora 缓存目录且指定了 transformers 缓存目录，则使用 transformers 缓存目录

        self.targets = targets
        self.loaders = {
            ref: LazyTensorLoader(
                ref.merged(
                    cache_dir=lora_cache_dir, trust_remote_code=trust_remote_code
                ).tensor_index(
                    cache_dir=transformers_cache_dir,
                ),
                lazy_unpickle=lazy_unpickle,
            )
            for ref in models
        }
        # 初始化加载器，用于加载每个模型的张量

        for model, loader in self.loaders.items():
            rules.procedural.append(
                LoadTensorRule(model, loader.index.tensor_paths, dtype=dtype)
            )
        # 为每个模型添加加载张量的过程性规则

        if operations is None:
            operations = {}
        self.operations = operations
        self.rules = rules
        self.cuda = cuda
        self.low_cpu_memory = low_cpu_memory
        # 初始化操作、规则集和配置选项

    def run(self, out_path: str, max_shard_size: int, clone_tensors: bool = False):
        """
        执行计算图并将结果保存到磁盘。

        此方法将按照计算图生成张量，并将每个张量保存到磁盘。张量计算安排以最小化内存使用。
        """
        writer = TensorWriter(out_path, max_shard_size=max_shard_size)
        # 初始化张量写入器，用于保存张量到指定路径

        for ref, tensor in tqdm.tqdm(self.generate_tensors(), total=len(self.targets)):
            # 遍历生成的张量及其引用

            if not self.low_cpu_memory:
                tensor = tensor.cpu()
            # 如果不是低内存模式，则将张量移至 CPU

            writer.save_tensor(ref.key, tensor, clone=clone_tensors)
            # 保存张量到磁盘

        writer.finalize()
        # 完成张量写入操作

    def generate_tensors(self) -> Iterator[Tuple[TensorReference, torch.Tensor]]:
        """
        生成指定的目标张量。

        构建计算图，安排执行，然后计算所有张量，并生成每个目标张量及其引用。根据最后一次使用保留或清除内存中的张量，以优化内存使用。
        """
        schedule = self._schedule_ops()
        # 生成执行张量操作的计划

        # 确定每个张量的最后使用时间，以便随后清除
        last_use = {}
        for idx, (component, _) in enumerate(schedule):
            for j in range(len(schedule) - 1, idx, -1):
                if component in schedule[j][1].inputs:
                    break
            last_use[component] = j

        tensors: Dict[TensorReference, torch.Tensor] = {}
        for idx, (component, operation) in enumerate(schedule):
            # 遍历计划中的每个操作

            tensor_args = {}
            for ref in operation.inputs:
                value = tensors[ref]
                if self.cuda and value.device.type != "cuda":
                    value = value.cuda()
                tensor_args[ref] = value
            # 准备操作的输入张量

            res = self._perform_operation(operation, tensor_args)
            # 执行操作

            del tensor_args
            # 删除输入张量，减少内存占用

            if res is not None:
                tensors[component] = res
            # 将结果张量保存

            if component in self.targets:
                yield (component, res)
            # 如果是目标张量，则生成

            # 清除不再使用的张量
            expired = []
            for key in tensors:
                if idx >= last_use[key]:
                    expired.append(key)

            for key in expired:
                del tensors[key]


    def _perform_operation(
        self, operation: Operation, tensor_args: Dict[TensorReference, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        使用提供的张量参数执行给定操作。

        Args:
            operation: 要执行的操作。
            tensor_args: 张量引用及其实际值的映射。
        """
        if operation.function == "load_tensor":
            return self._load_tensor(operation)
        # 如果操作是加载张量，则调用_load_tensor 方法

        if operation.function in self.operations:
            return self.operations[operation.function](
                input_tensors=tensor_args, **operation.kwargs
            )
        # 如果操作在操作字典中，则执行该操作

        raise RuntimeError(f"Unimplemented function {operation.function}")
        # 如果操作未实现，则抛出运行时错误


    def _load_tensor(self, operation: Operation):
        """从输入模型加载张量。"""
        assert operation.function == "load_tensor"
        # 确认操作是加载张量

        res = self.loaders[operation.kwargs["model"]].get_tensor(
            operation.kwargs["key"]
        )
        # 加载指定模型的张量

        if operation.kwargs["dtype"]:
            res = res.to(dtype=operation.kwargs["dtype"])
        # 转换张量数据类型

        if self.cuda and self.low_cpu_memory:
            res = res.cuda()
        # 如果使用 CUDA 并且是低内存模式，则将张量移至 GPU
        return res


    def _compare_key(self, ref: TensorReference):
        """
        生成用于排序计算的键。

        旨在最小化任何时刻必须驻留在内存中的分片数量。
        """
        if ref.model:
            shard_key = _normalized_shard_name(
                self.loaders[ref.model].index.tensor_paths[ref.key]
            )
        else:
            shard_key = ""
        # 生成排序键

        out_key = "" if ref in self.targets else "input"
        # 生成输出键

        return (out_key, shard_key, ref.key)
        # 返回用于排序的键


    def _schedule_ops(self) -> List[Tuple[TensorReference, Operation]]:
        """
        生成执行张量操作的计划。

        构建张量计算的依赖图，并以满足所有依赖的同时最小化内存使用的方式排序。
        """
        dependencies, ops = self._build_dependencies()
        # 构建依赖图和操作

        edge_tups = []
        for node in dependencies:
            for dependency in dependencies[node]:
                edge_tups.append((dependency, node))
        # 构建边的元组

        graph = networkx.DiGraph(edge_tups)
        res = list(
            networkx.lexicographical_topological_sort(graph, key=self._compare_key)
        )
        # 构建并排序计算图

        return [(r, ops[r]) for r in res]
        # 返回执行计划


    def _build_dependencies(
        self,
    ) -> Tuple[
        Dict[TensorReference, Set[TensorReference]], Dict[TensorReference, Operation]
    ]:
        """
        为计算构建依赖图，并选择规则来生成每个张量。
        """
        dependencies: Dict[TensorReference, Set[TensorReference]] = {}
        ops: Dict[TensorReference, Operation] = {}
        # 初始化依赖字典和操作字典

        def _visit(node: TensorReference):
            if node in ops:
                return
            # 如果节点已在操作字典中，则返回

            operation = self.rules.get(node)
            if not operation:
                raise RuntimeError(f"No rule to produce {node}")
            # 获取节点的操作，如果没有则抛出错误

            ops[node] = operation
            # 保存操作

            dependencies[node] = set()
            for dependency in operation.inputs:
                dependencies[node].add(dependency)
            # 设置依赖

            for dependency in operation.inputs:
                _visit(dependency)
            # 递归访问依赖节点

        for target in self.targets:
            _visit(target)
        # 访问每个目标张量

        return dependencies, ops
        # 返回依赖图和操作字典


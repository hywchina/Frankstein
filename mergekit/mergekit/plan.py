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
from typing import Any, Dict, List, Optional, Tuple

import torch

import mergekit.merge_methods as merge_methods
from mergekit.architecture import ArchitectureInfo
from mergekit.common import ModelReference
from mergekit.config import (
    ConfigReader,
    InputSliceDefinition,
    MergeConfiguration,
    OutputSliceDefinition,
)
from mergekit.graph import Operation, TensorReference
from mergekit.merge_methods import MergeMethod


def plan(
    merge_config: MergeConfiguration,
    arch_info: ArchitectureInfo,
    embed_permutations: Optional[Dict[ModelReference, torch.Tensor]] = None,
) -> Tuple[List[TensorReference], Dict[TensorReference, Operation]]:
    
    """
    函数逻辑总结：

    plan 函数首先初始化必要的数据结构，包括目标张量列表和操作规则字典。
    函数根据配置确定合并方法和基础模型。
    如果指定了要合并的模型列表，函数创建对应的输入片段定义。
    接着，函数为架构中定义的预处理和后处理权重创建相应的操作。
    对于配置中的每个输出片段，函数调用 plan_slice 来规划该片段内的层级操作。
    最后，函数返回包含所有目标张量引用和操作规则的元组。这些信息可用于执行模型合并操作。
    """
    # 定义一个名为 plan 的函数，用于规划模型合并操作

    layer_idx = 0
    # 初始化层索引

    targets = []
    rules = {}
    # 初始化目标张量引用列表和操作规则字典

    method = merge_methods.get(merge_config.merge_method)
    # 获取合并方法

    base_model = (
        ModelReference.parse(merge_config.base_model)
        if merge_config.base_model
        else None
    )
    # 解析基础模型的引用，如果有的话

    # 如果指定了要合并的模型而不是输出片段，则计算它们
    if merge_config.models:
        # 检查配置是否指定了模型而非输出片段

        if merge_config.slices:
            raise RuntimeError("Must specify either models to merge or output slices")
            # 如果同时指定了模型和输出片段，则抛出错误

        slices_in = []
        base_included = False
        # 初始化输入片段列表和基础模型是否包含的标记

        for model_in in merge_config.models:
            # 遍历要合并的模型

            mref = ModelReference.parse(model_in.model)
            # 解析每个模型的引用

            if base_model and mref == base_model:
                base_included = True
                # 检查是否包含基础模型

            model_cfg = mref.config()
            num_layers = arch_info.num_layers(model_cfg)
            # 获取模型配置和层数

            slices_in.append(
                InputSliceDefinition(
                    layer_range=[0, num_layers],
                    model=model_in.model,
                    parameters=model_in.parameters,
                )
            )
            # 为每个模型创建输入片段定义

        if base_model and not base_included:
            logging.info("Base model specified but not in input models - adding")
            base_cfg = base_model.config()
            num_layers = arch_info.num_layers(base_cfg)
            slices_in.append(
                InputSliceDefinition(
                    layer_range=[0, num_layers],
                    model=str(base_model),
                )
            )
            # 如果指定了基础模型但未包含在输入模型中，则添加之

        merge_config.slices = [OutputSliceDefinition(sources=slices_in)]
        merge_config.models = None
        # 更新配置中的片段定义并清空模型列表

    for weight_name in arch_info.pre_weights():
        # 遍历架构信息中定义的预处理权重

        is_embed = weight_name in arch_info.embed_weights()
        # 检查权重是否为嵌入权重

        tr, op = make_operation(
            merge_config,
            weight_name,
            merge_config.slices[0].sources,
            t=0,
            extra_kwargs={"embed_permutations": embed_permutations},
            function="merge_embed" if (is_embed and embed_permutations) else "merge",
        )
        targets.append(tr)
        rules[tr] = op
        # 为每个预处理权重创建操作并添加到规则中

    for section in merge_config.slices:
        # 遍历配置中的输出片段

        (new_targets, new_rules, new_layers) = plan_slice(
            config=merge_config,
            definition=section,
            arch_info=arch_info,
            layer_base=layer_idx,
            method=method,
        )
        targets.extend(new_targets)
        rules.update(new_rules)
        layer_idx += new_layers
        # 为每个片段规划操作并更新目标列表和规则字典

    for weight_name in arch_info.post_weights():
        # 遍历架构信息中定义的后处理权重

        is_embed = weight_name in arch_info.embed_weights()
        # 检查权重是否为嵌入权重

        tr, op = make_operation(
            merge_config,
            weight_name,
            merge_config.slices[-1].sources,
            t=1,
            extra_kwargs={"embed_permutations": embed_permutations},
            function="merge_embed" if (is_embed and embed_permutations) else "merge",
        )
        targets.append(tr)
        rules[tr] = op
        # 为每个后处理权重创建操作并添加到规则中

    return (targets, rules)
    # 返回目标列表和规则字典



def make_operation(
    config: MergeConfiguration,
    name_out: str,
    tensor_sources: List[InputSliceDefinition],
    t: float,
    names_in: Optional[List[str]] = None,
    sdef: Optional[OutputSliceDefinition] = None,
    extra_dependencies: Optional[List[TensorReference]] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
    function: str = "merge",
):
    """
    函数逻辑总结：

    make_operation 函数用于根据提供的配置和规则创建一个合并或处理张量的操作。
    函数首先检查是否提供了输入张量的名称，如果没有，就使用输出张量的名称作为默认输入名称。
    然后，函数初始化一个输入张量引用列表，并创建一个包含配置阅读器和其他参数的关键字参数字典。
    对于每个输入张量来源，函数解析模型引用并创建相应的张量引用，然后将这些引用添加到输入张量列表中。
    如果提供了额外的依赖张量，函数也将这些依赖添加到输入列表中。
    函数创建一个代表输出张量的引用，然后创建一个包含所有输入和参数的操作对象。
    最后，函数返回输出张量引用和创建的操作对象。这个操作对象可以用于执行定义的函数（如合并张量等）。
    """
    # 定义一个名为 make_operation 的函数，用于创建一个操作
    # 函数接受合并配置、输出张量名称、张量来源列表、浮点数 t、输入张量名称列表、输出片段定义、额外依赖列表、额外关键字参数和函数名称

    if names_in is None:
        names_in = [name_out] * len(tensor_sources)
    # 如果未提供输入张量名称，则默认为与输出张量名称相同

    input_tensors = []
    # 初始化输入张量列表

    kwargs = {
        "config": ConfigReader(
            config=config,
            tensor_name=name_out,
            t=t,
            slice_out=sdef,
            slices_in=tensor_sources,
        ),
        "parameter_name": name_out,
    }
    # 创建关键字参数字典，包含配置阅读器和参数名称

    if extra_kwargs:
        kwargs.update(extra_kwargs)
    # 如果有额外的关键字参数，更新到 kwargs 中

    for i, s in enumerate(tensor_sources):
        # 遍历张量来源

        input_tensors.append(
            TensorReference(model=ModelReference.parse(s.model), key=names_in[i])
        )
        # 对于每个来源，创建一个张量引用并添加到输入张量列表中

    if extra_dependencies:
        input_tensors.extend(extra_dependencies)
    # 如果有额外的依赖，将它们添加到输入张量列表中

    tr = TensorReference(model=None, key=name_out)
    # 创建一个输出张量引用

    op = Operation(function=function, inputs=input_tensors, kwargs=kwargs)
    # 创建一个操作，指定函数、输入张量和关键字参数

    return tr, op
    # 返回张量引用和操作



def plan_slice(
    config: MergeConfiguration,
    definition: OutputSliceDefinition,
    arch_info: ArchitectureInfo,
    layer_base: int,
    method: MergeMethod,
) -> Tuple[List[TensorReference], Dict[TensorReference, Operation], int]:
    """
    函数逻辑总结：
    plan_slice 函数首先调用 get_slice_indices 函数获取输出片段定义的分片指数。
    然后，它确定层数，即第一个分片指数列表的长度。
    函数初始化一个空的操作规则字典和一个空的目标张量引用列表。
    接下来，它遍历每一层，并计算一个比例因子 t。这个比例因子用于插值或分配权重，当片段只有一层时，t 为 1。
    函数对每一层调用 plan_layer 函数来规划该层的操作，并将这些操作添加到操作规则字典中。
    最后，函数返回目标张量引用列表、操作规则字典和层数。这些输出可以用于进一步处理或应用这些规划好的操作。
    """

    # 定义一个名为 plan_slice 的函数，接受合并配置、输出片段定义、架构信息、层基数和合并方法作为参数
    # 返回值是一个元组，包含张量引用列表、操作规则字典和整数表示的层数

    slice_indices = get_slice_indices(definition)
    # 获取输出片段定义的分片指数

    num_layers = len(slice_indices[0])
    # 计算层数，即第一个分片指数列表的长度

    rules = {}
    targets = []
    # 初始化操作规则字典和目标张量引用列表

    for idx in range(num_layers):
        # 遍历每一层

        if num_layers > 1:
            t = idx / (num_layers - 1)
        else:
            t = 1
        # 计算 t 值，用于插值或分配权重，当只有一层时 t 为 1

        plan_layer(
            config=config,
            definition=definition,
            arch_info=arch_info,
            layer_base=layer_base,
            slice_indices=slice_indices,
            method=method,
            rules=rules,
            targets=targets,
            idx=idx,
            t=t,
        )
        # 规划每一层的操作

    return targets, rules, num_layers
    # 返回目标张量引用列表、操作规则字典和层数


def plan_layer(
    config: MergeConfiguration,
    definition: OutputSliceDefinition,
    arch_info: ArchitectureInfo,
    layer_base: int,
    slice_indices: List[List[int]],
    method: MergeMethod,
    rules: Dict[TensorReference, Operation],
    targets: List[TensorReference],
    idx: int,
    t: float,
):
    """
    函数逻辑总结：

    plan_layer 函数用于根据提供的配置和定义来规划特定层的操作。
    函数首先获取合并方法的一般依赖关系。
    接着，它遍历输出片段定义中的所有源，获取每个源的层索引和模型引用，并将输入层的依赖关系添加到额外依赖列表中。
    然后，函数遍历架构信息中定义的层权重格式，并为输出层和每个源的输入层生成格式化的名称。
    对于每种层权重格式，函数调用 make_operation 来创建相应的操作，并将这些操作添加到规则字典中。
    这样，函数为每个层创建必要的操作并更新规则字典，以便这些操作能够在合并模型时被执行。
    """
    # 定义一个名为 plan_layer 的函数，它接受多个参数，包括合并配置、输出片段定义、架构信息、层基数、分片指数列表、合并方法、规则字典、目标列表、索引和浮点数 t

    extra_dependencies = list(method.general_dependencies())
    # 获取并列出合并方法的一般依赖关系

    for si, s in enumerate(definition.sources):
        # 遍历输出片段定义中的所有源

        source_layer_idx = slice_indices[si][idx]
        # 获取源的层索引

        source_model = ModelReference.parse(s.model)
        # 解析源的模型引用

        extra_dependencies.extend(
            method.input_layer_dependencies(source_model, source_layer_idx)
        )
        # 扩展额外依赖，包括输入层的依赖关系

    for name_format in arch_info.layer_weight_formats():
        # 遍历架构信息中定义的层权重格式

        name_out = name_format.format(idx=layer_base + idx)
        # 格式化输出名称，包含层基数和索引

        names_in = [
            name_format.format(idx=slice_indices[si][idx])
            for (si, _) in enumerate(definition.sources)
        ]
        # 为每个源生成格式化输入名称

        tr, op = make_operation(
            config,
            name_out,
            definition.sources,
            t,
            names_in=names_in,
            sdef=definition,
            extra_dependencies=extra_dependencies,
        )
        # 创建操作

        rules[tr] = op
        # 将创建的操作添加到规则字典中

        targets.append(tr)


def get_slice_indices(definition: OutputSliceDefinition):

    """
    函数逻辑总结：

    get_slice_indices 函数接收一个 OutputSliceDefinition 对象，该对象包含多个源输入片段的定义。
    函数遍历这些源输入片段的定义，每个定义包含一个层范围（layer_range），表示所需层的起始和结束索引。
    对于每个源输入片段，函数生成一个表示这个层范围的整数列表，并将其添加到一个总的分片指数列表中。
    在添加新的层索引列表之前，函数检查之前添加的层索引列表是否有相同的长度。如果不一致，说明输入片段的层数量不匹配，函数将抛出运行时错误。
    最终，函数返回包含所有输入片段层索引的总列表。
    
    """
    # 定义一个名为 get_slice_indices 的函数，接受一个 OutputSliceDefinition 对象作为参数

    slice_indices = []
    # 初始化一个空列表，用于存储分片指数

    for s in definition.sources:
        # 遍历输出片段定义中的源输入片段定义

        indices = list(range(s.layer_range[0], s.layer_range[1]))
        # 生成一个从 layer_range[0] 到 layer_range[1] 的整数列表，表示层的索引

        if slice_indices and len(indices) != len(slice_indices[-1]):
            # 如果已经有存储的分片指数，且当前索引列表的长度与最后一个存储的分片指数的长度不同

            raise RuntimeError(
                "All inputs to a slice must contain the same number of layers"
            )
            # 抛出运行时错误，指出所有输入到一个片段的层必须具有相同数量的层

        slice_indices.append(indices)
        # 将当前的索引列表添加到分片指数列表中

    return slice_indices
    # 返回分片指数列表


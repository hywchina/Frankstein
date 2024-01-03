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

from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel
from typing_extensions import TypeAlias

from mergekit.common import ModelReference

ScalarOrGradient: TypeAlias = Union[float, List[float]]


class ConditionalParameter(BaseModel):
    """
    类功能简述（ConditionalParameter）：

    ConditionalParameter 类用于定义带有条件的参数。
    value 属性存储参数的实际值，其类型可以是标量或梯度。
    filter 属性是一个可选的字符串，用于指定应用该参数的条件或过滤器。
    """
    # 定义一个名为ConditionalParameter的类，继承自BaseModel

    value: ScalarOrGradient
    # 类的一个属性，表示值，类型为ScalarOrGradient（标量或梯度）

    filter: Optional[str] = None
    # 类的另一个属性，表示过滤条件，是一个可选的字符串，默认为None



ParameterSetting: TypeAlias = Union[
    ConditionalParameter, List[ConditionalParameter], ScalarOrGradient
]
# 定义一个类型别名ParameterSetting，它可以是ConditionalParameter、ConditionalParameter的列表或ScalarOrGradient
"""
类型别名说明（ParameterSetting）：

ParameterSetting 是一个类型别名，它表示参数设置的类型。
这个类型可以是单个 ConditionalParameter 对象、ConditionalParameter 对象的列表，或者是 ScalarOrGradient 类型的值。
这种灵活的类型定义允许 ParameterSetting 表示多种形式的参数设置，包括单个值、具有特定条件的值，以及值的列表。

"""


def evaluate_setting(
    tensor_name: str, setting: ParameterSetting, t: float = 0
) -> float:
    
    """
    函数逻辑总结：

    evaluate_setting 函数用于根据提供的设置参数 setting，张量名称 tensor_name，以及一个浮点数 t 来计算并返回一个浮点数值。
    函数首先检查 setting 的类型，根据其类型进行不同的处理：
    如果 setting 是基本数据类型（如浮点数、整数、布尔值、字符串），则直接返回该值。
    如果 setting 是列表类型，则进一步检查列表元素的类型，并根据不同情况进行处理：
    如果所有元素都是数值型，则进行线性插值。
    如果所有元素都是基本数据类型，则根据 t 计算索引并返回相应的元素。
    如果元素包含更复杂的结构（如包含过滤条件的对象），则递归调用 evaluate_setting，根据过滤条件和 t 的值选择合适的元素。
    如果 setting 不是上述任何一种类型，则抛出一个错误。
    如果没有任何返回值，则最后返回 None。
    
    """
    # 定义一个名为 evaluate_setting 的函数，接受张量名称、设置参数和一个默认为0的浮点数 t

    if isinstance(setting, (float, int, bool, str)):
        # 如果设置参数是浮点数、整数、布尔值或字符串类型，则直接返回该设置
        return setting

    elif isinstance(setting, list):
        # 如果设置参数是列表类型

        if all(isinstance(e, (int, float)) for e in setting):
            # 如果列表中的所有元素都是整数或浮点数

            scaled = t * (len(setting) - 1)
            # 根据 t 值和列表长度计算一个缩放值

            i0 = int(scaled)
            # 将缩放值向下取整得到索引 i0

            i1 = min(len(setting) - 1, i0 + 1)
            # 得到另一个索引 i1，它是 i0 加 1 和列表最大索引中的较小者

            frac = scaled - i0
            # 计算 i0 与缩放值之间的差，作为一个分数

            return (1 - frac) * setting[i0] + frac * setting[i1]
            # 返回线性插值结果，即根据 frac 的值在 setting[i0] 和 setting[i1] 之间进行插值

        elif all(isinstance(e, (float, int, bool, str)) for e in setting):
            # 如果列表中的所有元素都是浮点数、整数、布尔值或字符串类型

            return setting[int(t * (len(setting) - 1))]
            # 根据 t 值和列表长度计算索引，并返回该索引处的元素

        else:
            # 如果列表中包含其他类型的元素

            for cond in setting:
                # 遍历列表中的每个元素

                if (
                    (cond.filter is None)
                    or (cond.filter == "*")
                    or cond.filter in tensor_name
                ):
                    # 如果元素没有过滤条件，或过滤条件为 '*'，或张量名称符合过滤条件

                    res = evaluate_setting(tensor_name, cond.value, t)
                    # 递归调用 evaluate_setting 函数

                    return res
            # 返回递归调用的结果

    else:
        # 如果设置参数不是上述任何一种类型

        raise RuntimeError(f"Unexpected setting value: {setting}")
        # 抛出运行时错误，提示设置值的类型不符合预期

    return None
    # 如果函数中途没有返回任何值，则最后返回 None


class InputSliceDefinition(BaseModel):
    """
    类功能简述（InputSliceDefinition）：

    InputSliceDefinition 类用于定义一个输入片段。
    model 属性指定了输入片段所使用的模型的名称。
    layer_range 属性定义了从模型中使用的层的范围，这可以是模型的一部分，如特定层或层的一段。
    parameters 属性是一个字典，它允许为输入片段指定各种参数设置，这些设置可能会影响片段的处理或功能。
    """
    # 定义一个名为InputSliceDefinition的类，继承自BaseModel

    model: str
    # 类的属性之一，表示模型的名称，是一个字符串

    layer_range: Tuple[int, int]
    # 类的另一个属性，表示层范围，是一个由两个整数组成的元组

    parameters: Optional[Dict[str, ParameterSetting]] = None
    # 类的另一个属性，表示参数，是一个可选的字典，默认为None。字典的键是字符串，值是ParameterSetting类型



class InputModelDefinition(BaseModel):

    """
    类功能简述（InputModelDefinition）：

    InputModelDefinition 类用于定义一个输入模型。
    model 属性指定了输入模型的名称。
    parameters 属性是一个字典，它允许为输入模型指定各种参数设置，这些设置可能会影响模型的处理或功能。
    """
    # 定义一个名为InputModelDefinition的类，继承自BaseModel

    model: str
    # 类的属性之一，表示模型的名称，是一个字符串

    parameters: Optional[Dict[str, ParameterSetting]] = None
    # 类的另一个属性，表示参数，是一个可选的字典，默认为None。字典的键是字符串，值是ParameterSetting类型



class OutputSliceDefinition(BaseModel):
    """
    OutputSliceDefinition 类用于定义一个输出片段。每个输出片段包含若干输入片段（通过sources属性定义），并可能关联一个基础模型（base_model），指定一个残差权重（residual_weight），以及包含一系列参数设置（parameters）。
    sources 属性是输入片段定义的列表，其中每个输入片段可能包含不同的模型输出或处理逻辑。
    base_model 属性是一个可选属性，允许为输出片段指定一个基础模型。
    residual_weight 属性用于控制残差连接的权重，这在合并不同模型输出时可能会用到。
    parameters 属性是一个字典，允许为输出片段指定各种参数设置，这些设置可能会影响片段的处理或合并方式。
    """
    # 定义一个名为OutputSliceDefinition的类，继承自BaseModel

    sources: List[InputSliceDefinition]
    # 类的属性之一，表示源输入片段定义的列表

    base_model: Optional[str] = None
    # 类的另一个属性，表示基础模型，是一个可选的字符串，默认为None

    residual_weight: Optional[float] = None
    # 类的另一个属性，表示残差权重，是一个可选的浮点数，默认为None

    parameters: Optional[Dict[str, ParameterSetting]] = None
    # 类的另一个属性，表示参数，是一个可选的字典，默认为None。字典的键是字符串，值是ParameterSetting类型



class MergeConfiguration(BaseModel):

    """
    方法逻辑总结：

    referenced_models 方法：
        创建一个空集合用于存储模型引用。
        如果有基础模型，解析并添加到集合中。
        遍历输入模型参数，解析每个键作为模型引用并添加到集合中。
        如果有输入模型定义，解析每个定义并添加到集合中。
        如果有输出片段定义，解析每个片段定义中的源作为模型引用并添加到集合中。
        返回模型引用的列表。

    validate 方法：
        检查是否至少定义了输出片段或输入模型中的一个。
        如果没有定义输出片段也没有定义输入模型，或者同时定义了这两者，抛出一个运行时错误，指出必须指定输出片段或输入模型以进行合并。
            
    """
    # 定义一个名为MergeConfiguration的类，继承自BaseModel

    merge_method: str
    slices: Optional[List[OutputSliceDefinition]] = None
    models: Optional[List[InputModelDefinition]] = None
    input_model_parameters: Dict[str, Dict[str, ParameterSetting]] = None
    parameters: Optional[Dict[str, ParameterSetting]] = None
    base_model: Optional[str] = None
    dtype: Optional[str] = None
    tokenizer_source: Optional[str] = None
    # 类的属性：合并方法、输出片段定义列表、输入模型定义列表、输入模型参数、一般参数、基础模型、数据类型和分词器源

    def referenced_models(self) -> List[ModelReference]:
        # 定义一个名为referenced_models的方法，返回ModelReference对象的列表

        models = set()
        # 创建一个空集合用于存储模型引用

        if self.base_model:
            # 如果配置中有基础模型

            models.add(ModelReference.parse(self.base_model))
            # 解析基础模型并添加到集合中

        if self.input_model_parameters:
            # 如果配置中有输入模型参数

            for key in self.input_model_parameters:
                # 遍历输入模型参数中的键

                models.add(ModelReference.parse(key))
                # 解析每个键作为模型引用并添加到集合中

        if self.models:
            # 如果配置中有输入模型定义

            for model_in in self.models:
                # 遍历输入模型定义

                models.add(ModelReference.parse(model_in.model))
                # 解析每个模型定义并添加到集合中

        if self.slices:
            # 如果配置中有输出片段定义

            for s in self.slices:
                # 遍历输出片段定义

                for src in s.sources:
                    # 遍历每个片段定义中的源

                    models.add(ModelReference.parse(src.model))
                    # 解析每个源作为模型引用并添加到集合中

        return list(models)
        # 返回模型引用集合的列表形式

    def validate(self):
        # 定义一个名为validate的方法，用于验证配置的有效性

        if ((not self.slices) and (not self.models)) or (self.slices and self.models):
            # 如果没有定义输出片段也没有定义输入模型，或者同时定义了输出片段和输入模型

            raise RuntimeError("Must specify either output slices or models to merge")
            # 抛出运行时错误，指出必须指定输出片段或输入模型以进行合并



class ConfigReader(BaseModel):

    """
    方法逻辑总结：

    base_model 属性：

        检查是否有输出片段定义(slice_out)以及它是否包含基础模型(base_model)。
        如果有，返回该基础模型。
        否则，返回配置(config)中的基础模型。
        如果两者都不存在，返回None。
    parameter 方法：
        接收参数名称、模型引用、默认值和是否为必需参数。
        首先检查输入片段定义(slices_in)，如果找到匹配的模型并且其中包含所需参数，返回该参数的值。
        然后检查输出片段定义(slice_out)，如果其中包含所需参数，返回该参数的值。
        接着检查配置(config)中的输入模型参数和一般参数，如果找到所需参数，返回该参数的值。
        如果参数是必需的，但未找到，则抛出运行时错误。
        如果未找到参数值，则返回默认值。
    """
    # 定义一个名为ConfigReader的类，继承自BaseModel

    config: MergeConfiguration
    tensor_name: str
    t: float
    slice_out: Optional[OutputSliceDefinition]
    slices_in: Optional[List[InputSliceDefinition]]
    # 类的属性：配置对象、张量名称、浮点数t、输出片段定义和输入片段定义列表

    @property
    def base_model(self) -> Optional[ModelReference]:
        # 定义一个名为base_model的属性，返回一个可选的ModelReference对象

        if self.slice_out and self.slice_out.base_model:
            # 如果有输出片段定义且其有基础模型定义

            res = self.slice_out.base_model
            # 获取输出片段定义中的基础模型

        else:
            res = self.config.base_model
            # 否则，获取配置中的基础模型

        if res:
            # 如果找到了基础模型

            return ModelReference.parse(res)
            # 返回解析后的ModelReference对象

        return None
        # 如果没有找到基础模型，返回None

    def parameter(
        self,
        name: str,
        model: Optional[ModelReference] = None,
        default: Any = None,
        required: bool = False,
    ) -> Any:
        # 定义一个名为parameter的方法，接收参数名称、模型引用、默认值和是否必需的标志

        if model and self.slices_in:
            # 如果提供了模型引用且有输入片段定义

            for s in self.slices_in:
                # 遍历输入片段定义

                if s.model == str(model) and s.parameters and name in s.parameters:
                    # 如果片段的模型匹配且包含所需参数

                    value = evaluate_setting(
                        self.tensor_name, s.parameters[name], self.t
                    )
                    # 计算参数值

                    if value is not None:
                        return value
                    # 如果参数值存在，返回该值

        if self.slice_out:
            # 如果有输出片段定义

            if self.slice_out.parameters and name in self.slice_out.parameters:
                # 如果输出片段定义包含所需参数

                value = evaluate_setting(
                    self.tensor_name, self.slice_out.parameters[name], self.t
                )
                # 计算参数值

                if value is not None:
                    return value
                # 如果参数值存在，返回该值

        if (
            self.config.input_model_parameters
            and model
            and str(model) in self.config.input_model_parameters
        ):
            # 如果配置中包含输入模型参数且给定了模型引用

            if name in self.config.input_model_parameters[self.slice_in.model]:
                # 如果输入模型参数中包含所需参数

                value = evaluate_setting(
                    self.tensor_name,
                    self.config.input_model_parameters[str(model)][name],
                    self.t,
                )
                # 计算参数值

                if value is not None:
                    return value
                # 如果参数值存在，返回该值

        if self.config.parameters and name in self.config.parameters:
            # 如果配置中包含所需参数

            value = evaluate_setting(
                self.tensor_name,
                self.config.parameters[name],
                self.t,
            )
            # 计算参数值

            if value is not None:
                return value
            # 如果参数值存在，返回该值

        if required:
            # 如果参数是必需的

            suffix = (
                f" for {str(model)}.{self.tensor_name}"
                if model
                else f" for {self.tensor_name}"
            )
            # 构建错误信息的后缀部分

            raise RuntimeError(f"Missing required parameter {name}{suffix}")
            # 抛出运行时错误，指出缺少必需参数

        return default
        # 如果未找到参数值，返回默认值

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
from typing import List, Optional

from pydantic import BaseModel
from transformers import PretrainedConfig


class ArchitectureInfo(ABC):
    @abstractmethod
    def pre_weights(self) -> List[str]:
        """Return a list of all weights preceding the first layer."""
        ...
        # 定义一个抽象方法 pre_weights。这个方法应当返回一个字符串列表，包含第一层之前的所有权重。

    @abstractmethod
    def post_weights(self) -> List[str]:
        """Return a list of all weights following the final layer."""
        ...
        # 定义一个抽象方法 post_weights。这个方法应当返回一个字符串列表，包含最后一层之后的所有权重。

    @abstractmethod
    def layer_weight_formats(self) -> List[str]:
        """Return a list of format strings all weights associated with a layer."""
        ...
        # 定义一个抽象方法 layer_weight_formats。这个方法应当返回一个字符串列表，包含与一层相关的所有权重的格式字符串。

    @abstractmethod
    def embed_weights(self) -> List[str]:
        ...
        # 定义一个抽象方法 embed_weights。具体实现应当返回与嵌入相关的权重列表。

    def num_layers(self, config: PretrainedConfig) -> int:
        return config.num_hidden_layers
        # 定义一个方法 num_layers。这个方法返回模型隐藏层的数量。

    def num_layers_config_key(self) -> str:
        """Key in config that represents number of layers"""
        return "num_hidden_layers"
        # 定义一个方法 num_layers_config_key。这个方法返回表示层数的配置键值。


class StaticTensorNames(ArchitectureInfo, BaseModel, frozen=True):
    # 定义一个名为 StaticTensorNames 的类，继承自 ArchitectureInfo 和 BaseModel。
    # 参数 frozen=True 表示这个模型是不可变的（即实例化后不能更改属性值）。

    # 架构的名称。
    name: str

    pre_weight_names: List[str]  # weights applied before first layer # 在第一层之前应用的权重名称列表。
    post_weight_names: List[str]  # weights applied after last layer # 在最后一层之后应用的权重名称列表。
    embed_weight_names: List[str]  # weights for embed/lm_head # 用于嵌入（embed）或语言模型头（lm_head）的权重名称列表。
    layer_prefix_format: str # 层的前缀格式。
    layer_weight_suffixes: List[str] # 层相关的权重后缀列表。
    num_layers_key: Optional[str] = None # 代表层数的键名，可选。

    # 重写 pre_weights 方法，返回 pre_weight_names 列表。
    def pre_weights(self) -> List[str]:
        return self.pre_weight_names

    # 重写 post_weights 方法，返回 post_weight_names 列表。
    def post_weights(self) -> List[str]:
        return self.post_weight_names

    # 重写 embed_weights 方法，返回 embed_weight_names 列表。
    def embed_weights(self) -> List[str]:
        return self.embed_weight_names

    # 重写 layer_weight_formats 方法，返回层权重格式列表。
    def layer_weight_formats(self) -> List[str]:
        res = []
        for suffix in self.layer_weight_suffixes:
            res.append(self.layer_prefix_format + "." + suffix)
        return res
    
    # 重写 num_layers_config_key 方法，如果 num_layers_key 存在，返回它；否则，调用父类方法。
    def num_layers_config_key(self) -> str:
        if self.num_layers_key:
            return self.num_layers_key
        return super().num_layers_config_key()
    
    # 重写 num_layers 方法，返回由 num_layers_config_key 指定的配置项的值。
    def num_layers(self, config: PretrainedConfig) -> int:
        return getattr(config, self.num_layers_config_key())


LLAMA_INFO = StaticTensorNames(
    name="LlamaForCausalLM",
    pre_weight_names=["model.embed_tokens.weight"],
    post_weight_names=["model.norm.weight", "lm_head.weight"],
    embed_weight_names=["model.embed_tokens.weight", "lm_head.weight"],
    layer_prefix_format="model.layers.{idx}",
    layer_weight_suffixes=[
        "input_layernorm.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "mlp.gate_proj.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
    ],
)

MISTRAL_INFO = StaticTensorNames(
    name="MistralForCausalLM",
    # lol
    **LLAMA_INFO.model_dump(exclude=["name"]),
)


STABLELM_INFO = StaticTensorNames(
    name="StableLMEpochForCausalLM",
    post_weight_names=LLAMA_INFO.post_weight_names + ["model.norm.bias"],
    layer_weight_suffixes=LLAMA_INFO.layer_weight_suffixes
    + [
        "input_layernorm.bias",
        "post_attention_layernorm.bias",
    ],
    **LLAMA_INFO.model_dump(
        exclude=["name", "layer_weight_suffixes", "post_weight_names"]
    ),
)

GPT_NEOX_INFO = StaticTensorNames(
    name="GPTNeoXForCausalLM",
    pre_weight_names=["gpt_neox.embed_in.weight"],
    post_weight_names=[
        "gpt_neox.final_layer_norm.bias",
        "gpt_neox.final_layer_norm.weight",
        "embed_out.weight",
    ],
    embed_weight_names=["gpt_neox.embed_in.weight", "embed_out.weight"],
    layer_prefix_format="gpt_neox.layers.{idx}",
    layer_weight_suffixes=sum(
        (
            [f"{prefix}.weight", f"{prefix}.bias"]
            for prefix in [
                "attention.dense",
                "attention.query_key_value",
                "input_layernorm",
                "mlp.dense_4h_to_h",
                "mlp.dense_h_to_4h",
                "post_attention_layernorm",
            ]
        ),
        start=[],
    )
    + ["attention.bias", "attention.masked_bias", "attention.rotary_emb.inv_freq"],
)

GPT2_INFO = StaticTensorNames(
    name="GPT2LMHeadModel",
    pre_weight_names=["wte.weight", "wpe.weight"],
    post_weight_names=["ln_f.weight", "ln_f.bias"],
    embed_weight_names=["wte.weight"],
    layer_prefix_format="h.{idx}",
    layer_weight_suffixes=[
        "attn.c_attn.weight",
        "attn.c_attn.bias",
        "attn.c_proj.weight",
        "attn.c_proj.bias",
        "ln_1.weight",
        "ln_1.bias",
        "ln_2.weight",
        "ln_2.bias",
        "mlp.c_proj.weight",
        "mlp.c_proj.bias",
        "mlp.c_fc.weight",
        "mlp.c_fc.bias",
        "mlp.c_proj.weight",
        "mlp.c_proj.bias",
    ],
    num_layers_key="n_layer",
)

GPT2_SEQCLASS_INFO = StaticTensorNames(
    name="GPT2ForSequenceClassification",
    pre_weight_names=["transformer.wte.weight", "transformer.wpe.weight"],
    post_weight_names=[
        "transformer.ln_f.weight",
        "transformer.ln_f.bias",
        "score.weight",
    ],
    layer_prefix_format="transformer.h.{idx}",
    embed_weight_names=GPT2_INFO.embed_weight_names,
    layer_weight_suffixes=GPT2_INFO.layer_weight_suffixes,
    num_layers_key=GPT2_INFO.num_layers_key,
)


QWEN_INFO = StaticTensorNames(
    name="QWenLMHeadModel",
    pre_weight_names=["transformer.wte.weight"],
    post_weight_names=["transformer.ln_f.weight", "lm_head.weight"],
    embed_weight_names=["transformer.wte.weight", "lm_head.weight"],
    layer_prefix_format="transformer.h.{idx}",
    layer_weight_suffixes=[
        "attn.c_attn.bias",
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "ln_1.weight",
        "ln_2.weight",
        "mlp.c_proj.weight",
        "mlp.w1.weight",
        "mlp.w2.weight",
    ],
)

CHATGLM_INFO = StaticTensorNames(
    name="ChatGLMModel",
    pre_weight_names=[
        "transformer.embedding.word_embeddings.weight",
        "transformer.rotary_pos_emb.inv_freq",
    ],
    post_weight_names=[
        "transformer.encoder.final_layernorm.weight",
        "transformer.output_layer.weight",
    ],
    embed_weight_names=[
        "transformer.embedding.word_embeddings.weight",
        "transformer.output_layer.weight",
    ],
    layer_prefix_format="transformer.encoder.layers.{idx}",
    layer_weight_suffixes=[
        "input_layernorm.weight",
        "mlp.dense_4h_to_h.weight",
        "mlp.dense_h_to_4h.weight",
        "post_attention_layernorm.weight",
        "self_attention.dense.weight",
        "self_attention.query_key_value.bias",
        "self_attention.query_key_value.weight",
    ],
)

class PhiTensorNames(ArchitectureInfo):
    """
    类功能简述：

    PhiTensorNames 类为 MixFormerSequentialForCausalLM 架构提供必要的架构信息。
    它定义了如何获取该架构特有的预处理权重、后处理权重、嵌入权重以及层权重的名称和格式。
    类的方法包括获取预处理和后处理层的权重名称、嵌入层的权重名称、层权重的格式以及模型的层数和层数配置键。
    这些信息对于管理和操作此架构的模型特别重要，尤其是在模型合并、权重转换和模型重建等任务中。
    """

    # 继承自 ArchitectureInfo 的 PhiTensorNames 类，专用于 MixFormerSequentialForCausalLM 架构

    architecture_name: str = "MixFormerSequentialForCausalLM"
    # 类属性，定义架构名称

    def __init__(self, config: PretrainedConfig):
        self.config = config
        # 构造函数，接收并存储预训练配置

    def pre_weights(self) -> List[str]:
        # 定义预处理权重名称的方法
        return ["layers.0.wte.weight"]
        # 返回预处理层的权重名称列表

    def post_weights(self) -> List[str]:
        # 定义后处理权重名称的方法
        fake_layer_idx = self.config.n_layer + 1
        # 计算假设的层索引

        return [
            f"layers.{fake_layer_idx}.{suffix}"
            for suffix in ["linear.bias", "linear.weight", "ln.bias", "ln.weight"]
        ]
        # 返回后处理层的权重名称列表

    def embed_weights(self) -> List[str]:
        # 定义嵌入权重名称的方法
        fake_layer_idx = self.config.n_layer + 1
        # 计算假设的层索引

        return [
            "layers.0.wte.weight",
            f"layers.{fake_layer_idx}.linear.weight",
            f"layers.{fake_layer_idx}.linear.bias",
        ]
        # 返回嵌入层的权重名称列表

    def layer_weight_formats(self) -> List[str]:
        # 定义层权重格式的方法
        return [
            ("layers.{idx}." + suffix)
            for suffix in [
                "ln.bias",
                "ln.weight",
                "mixer.Wqkv.bias",
                "mixer.Wqkv.weight",
                "mixer.out_proj.bias",
                "mixer.out_proj.weight",
                "mixer.rotary_emb.inv_freq",
                "mlp.fc1.bias",
                "mlp.fc1.weight",
                "mlp.fc2.bias",
                "mlp.fc2.weight",
            ]
        ]
        # 返回层权重的格式列表

    def num_layers(self, config: PretrainedConfig) -> int:
        # 定义获取层数的方法
        return config.n_layer
        # 返回配置中定义的层数

    def num_layers_config_key(self) -> str:
        # 定义获取层数配置键的方法
        return "n_layer"
        # 返回层数配置键



def get_architecture_info(config: PretrainedConfig) -> StaticTensorNames:
    if len(config.architectures) != 1:
        raise RuntimeError("More than one architecture in config?")

    arch_name = config.architectures[0]
    if arch_name == PhiTensorNames.architecture_name:
        return PhiTensorNames(config)

     # 定义一个支持的架构列表。
    supported = [
        LLAMA_INFO,
        MISTRAL_INFO,
        GPT_NEOX_INFO,
        QWEN_INFO,
        GPT2_INFO,
        GPT2_SEQCLASS_INFO,
        CHATGLM_INFO,
        STABLELM_INFO,
    ]
    # 遍历支持的架构列表，检查是否有与给定架构名称匹配的架构。如果找到匹配的架构，则返回相应的信息对象。
    for arch in supported:
        if arch.name == arch_name:
            return arch

    raise RuntimeError(f"Unsupported architecture {arch_name}")

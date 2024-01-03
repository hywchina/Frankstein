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
from typing import Optional

import torch
import transformers
from pydantic import BaseModel

from mergekit import merge_methods
from mergekit.architecture import get_architecture_info
from mergekit.common import ModelReference, parse_kmb
from mergekit.config import MergeConfiguration
from mergekit.graph import Executor, RuleSet
from mergekit.plan import plan
from mergekit.tokenizer import build_tokenizer


class MergeOptions(BaseModel):
    """
    类功能简述：

    MergeOptions 类用于配置模型合并过程中的不同选项。
    它包含多个属性，如是否允许混合不同架构的模型、是否使用 CUDA、输出分片的大小、是否复制分词器等。
    这些选项用于控制合并过程中的行为，例如内存使用、性能优化、安全性和兼容性等方面。
    """
    # 定义一个名为 MergeOptions 的类，继承自 BaseModel

    allow_crimes: bool = False
    # 类的一个属性，表示是否允许“犯罪”（如混合不同架构的模型），默认为 False

    transformers_cache: Optional[str] = None
    # 类的一个属性，表示 transformers 缓存的路径，可选，默认为 None

    lora_merge_cache: Optional[str] = None
    # 类的一个属性，表示 LoRA 合并缓存的路径，可选，默认为 None

    cuda: bool = False
    # 类的一个属性，表示是否使用 CUDA（GPU 加速），默认为 False

    low_cpu_memory: bool = False
    # 类的一个属性，表示是否在 CPU 上使用低内存模式，有利于减少内存占用，但可能降低性能，默认为 False

    out_shard_size: int = parse_kmb("5B")
    # 类的一个属性，表示输出分片的大小，默认为解析“5B”得到的整数值

    copy_tokenizer: bool = True
    # 类的一个属性，表示是否复制分词器到输出路径，默认为 True

    clone_tensors: bool = False
    # 类的一个属性，表示是否克隆张量，默认为 False

    trust_remote_code: bool = False
    # 类的一个属性，表示是否信任远程代码，默认为 False

    random_seed: Optional[int] = None
    # 类的一个属性，表示随机种子，可选，默认为 None

    lazy_unpickle: bool = False
    # 类的一个属性，表示是否使用懒解压缩，默认为 False



def run_merge(merge_config: MergeConfiguration, out_path: str, options: MergeOptions):

    """
    函数逻辑总结：

    run_merge 函数首先设置数据类型、随机种子（如果提供）并检查输出需求。
    然后根据配置获取合并方法和架构信息，检查是否允许混合不同架构的模型。
    如果配置中指定了分词器源，则构建并保存分词器。
    函数接着规划合并操作，创建规则集和执行器。
    接下来执行合并操作，生成新的模型。
    函数尝试设置输出模型配置的词汇表大小和层数，然后保存配置。
    最后，如果需要，函数将尝试复制分词器。
    """

    # 定义一个名为 run_merge 的函数，用于执行模型合并

    dtype: Optional[torch.dtype] = {
        None: None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[merge_config.dtype]
    # 根据配置设置数据类型

    if options.random_seed is not None:
        transformers.trainer_utils.set_seed(options.random_seed)
    # 如果指定了随机种子，设置随机种子

    if not merge_config.models and not merge_config.slices:
        raise RuntimeError("No output requested")
    # 如果没有指定要合并的模型或输出片段，则抛出错误

    method = merge_methods.get(merge_config.merge_method)
    # 获取合并方法

    model_arch_info = [
        get_architecture_info(m.config()) for m in merge_config.referenced_models()
    ]
    # 获取每个参考模型的架构信息

    if not options.allow_crimes:
        if not all(a == model_arch_info[0] for a in model_arch_info[1:]):
            raise RuntimeError(
                "Must specify --allow-crimes to attempt to mix different architectures"
            )
    # 如果不允许混合不同架构的模型，检查所有模型是否具有相同的架构

    arch_info = model_arch_info[0]
    # 获取架构信息

    if merge_config.tokenizer_source:
        tokenizer, embed_permutations = build_tokenizer(
            merge_config, trust_remote_code=options.trust_remote_code
        )
        tokenizer.save_pretrained(out_path, safe_serialization=True)
    else:
        tokenizer = None
        embed_permutations = None
    # 构建并保存分词器，如果指定了分词器源

    (targets, static_rules) = plan(
        merge_config, arch_info, embed_permutations=embed_permutations
    )
    # 规划合并操作

    rules = RuleSet(static_rules)
    # 创建规则集

    exec = Executor(
        merge_config.referenced_models(),
        targets,
        rules,
        {"merge": method, "merge_embed": merge_methods.TokenizerPermutationMerge()},
        transformers_cache_dir=options.transformers_cache,
        lora_cache_dir=options.lora_merge_cache,
        dtype=dtype,
        cuda=options.cuda,
        low_cpu_memory=options.low_cpu_memory,
        trust_remote_code=options.trust_remote_code,
        lazy_unpickle=options.lazy_unpickle,
    )
    # 创建执行器

    exec.run(
        out_path,
        max_shard_size=options.out_shard_size,
        clone_tensors=options.clone_tensors,
    )
    # 执行合并操作

    cfg_out = method.model_out_config(merge_config)
    # 获取输出模型的配置

    if tokenizer:
        try:
            cfg_out.vocab_size = len(tokenizer.get_vocab())
        except Exception as e:
            logging.warning(
                "Unable to set vocabulary size in output config - you may need to manually correct it.",
                exc_info=e,
            )
    # 尝试设置输出配置的词汇表大小

    try:
        num_layers = sum(
            s.sources[0].layer_range[1] - s.sources[0].layer_range[0]
            for s in merge_config.slices
        )
        setattr(cfg_out, arch_info.num_layers_config_key(), num_layers)
    except Exception as e:
        logging.warning(
            "Unable to set number of layers in output config - you may need to manually correct it.",
            exc_info=e,
        )
    # 尝试设置输出配置的层数

    cfg_out.save_pretrained(out_path)
    # 保存输出模型的配置

    if options.copy_tokenizer and tokenizer is None:
        # 如果指定了复制分词器但没有分词器对象
        try:
            # 尝试复制分词器
            donor_model = merge_config.base_model
            if donor_model:
                donor_model = ModelReference.parse(donor_model)
            if not donor_model:
                donor_model = merge_config.referenced_models()[0]

            transformers.AutoTokenizer.from_pretrained(
                donor_model.path
            ).save_pretrained(out_path, safe_serialization=True)
        except Exception as e:
            logging.error(
                "Failed to save tokenizer. The merge was still successful, just copy it from somewhere else.",
                exc_info=e,
            )
    # 如果分词器不存在且指定了复制分词器选项，从基础模型或第一个参考模型复制分词器

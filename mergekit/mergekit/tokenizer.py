import logging
from typing import Dict, Optional, Tuple

import torch
import tqdm
import transformers

from mergekit.common import ModelReference
from mergekit.config import MergeConfiguration


def get_vocab_size(model_path: str, trust_remote_code: bool) -> Optional[int]:

    """
    函数逻辑总结：
    get_vocab_size 函数尝试使用 transformers 库的 AutoConfig.from_pretrained 方法来加载指定路径上的预训练模型配置。
    如果加载成功，它从加载的配置中提取并返回 vocab_size（词汇表大小）。
    如果在加载配置或提取词汇表大小的过程中发生异常，函数捕获这个异常并通过 logging 模块记录警告信息。
    如果无法获取词汇表大小，函数返回 None。
    """
    # 定义一个名为 get_vocab_size 的函数，接受模型路径和一个布尔值 trust_remote_code，返回可选的整数

    try:
        # 尝试执行以下代码块

        cfg = transformers.AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        # 使用 transformers 库从预训练模型中获取配置，model_path 是模型的路径，trust_remote_code 指示是否信任远程代码

        return cfg.vocab_size
        # 返回从配置中获取的词汇表大小

    except Exception as e:
        # 如果在尝试过程中发生任何异常

        logging.warning(f"Unable to get vocab size for {model_path}", exc_info=e)
        # 使用 logging 模块记录警告信息，指出无法获取指定路径模型的词汇表大小，并记录异常信息

    return None
    # 如果无法获取词汇表大小，返回 None


def build_tokenizer(
    config: MergeConfiguration,
    trust_remote_code: bool,
) -> Tuple[transformers.PreTrainedTokenizer, Dict[ModelReference, torch.IntTensor]]:
    
    """
    函数逻辑总结：

    函数首先确定基础模型，并从其加载一个预训练的分词器。
    然后，它遍历配置中引用的所有模型，尝试加载它们的分词器，并构建一个包含各模型词汇表的字典。
    接下来，根据配置中指定的源（基础模型、词汇表联合或特定模型），构建最终的分词器词汇表。
    最后，函数计算每个模型的词汇表到最终分词器词汇表的映射，以及处理可能的索引超出范围的情况。
    函数返回最终构建的分词器和词汇表映射字典。
    """
    # 定义一个名为 build_tokenizer 的函数，接受一个 MergeConfiguration 对象和一个布尔值 trust_remote_code
    # 返回值是一个元组，包含一个预训练分词器和一个从模型引用到 PyTorch 整数张量的字典

    base_model = None
    if config.base_model:
        base_model = ModelReference.parse(config.base_model)
    # 如果配置中指定了基础模型，解析并获取该模型

    if base_model is None:
        base_model = config.referenced_models()[0]
    # 如果没有基础模型，使用配置中引用的第一个模型

    if base_model is None:
        raise RuntimeError("No models referenced")
    # 如果没有引用任何模型，抛出运行时错误

    tokenizer_out = transformers.AutoTokenizer.from_pretrained(
        base_model.path, trust_remote_code=trust_remote_code
    )
    # 从预训练的基础模型加载分词器

    # 加载所有分词器
    logging.info("Loading tokenizers")
    vocabularies = {base_model: tokenizer_out.get_vocab()}
    # 初始化包含基础模型分词器词汇表的字典

    for model in config.referenced_models():
        if model == base_model:
            continue
        # 遍历配置中引用的所有模型，跳过基础模型

        try:
            model_tok = transformers.AutoTokenizer.from_pretrained(
                model.path, trust_remote_code=trust_remote_code
            )
            # 尝试加载每个模型的分词器
        except Exception:
            logging.warning(
                f"Unable to load tokenizer for {model}. Assuming same as {base_model}."
            )
            # 如果加载失败，记录警告并假设该模型使用与基础模型相同的分词器
            continue
        vocabularies[model] = model_tok.get_vocab()
        # 将加载的分词器词汇表添加到词汇表字典中

    logging.info("Building output tokenizer")
    # 构建最终的分词器词汇表
    if config.tokenizer_source == "base":
        pass
        # 如果配置指定使用基础分词器的词汇表，无需进一步操作
    elif config.tokenizer_source == "union":
        added = set(tokenizer_out.get_vocab().keys())
        # 如果配置指定合并所有分词器的词汇表

        for model_vocab in tqdm.tqdm(vocabularies.values(), total=len(vocabularies)):
            for tok in tqdm.tqdm(model_vocab, leave=False):
                if tok not in added:
                    tokenizer_out.add_tokens(tok)
                    added.add(tok)
        # 遍历每个分词器的词汇表，将新的词汇添加到基础分词器中

        del added
    elif config.tokenizer_source.startswith("model:"):
        tokenizer_out = transformers.AutoTokenizer.from_pretrained(
            config.tokenizer_source.removeprefix("model:"),
            trust_remote_code=trust_remote_code,
        )
        # 如果配置指定使用特定模型的分词器

    else:
        raise RuntimeError(f"Unimplemented tokenizer source: {config.tokenizer_source}")
        # 如果配置的分词器源是未实现的类型，抛出运行时错误

    vocab_out = tokenizer_out.get_vocab()
    # 获取最终分词器的词汇表

    logging.info("Building permutations")
    permutations = {}
    # 初始化存储词汇表映射的字典

    for model in tqdm.tqdm(config.referenced_models()):
        if model in vocabularies:
            model_vocab = vocabularies[model]
        else:
            model_vocab = vocabularies[base_model]
        # 获取每个模型的词汇表

        vocab_size = get_vocab_size(model, trust_remote_code=trust_remote_code)
        # 获取每个模型的词汇表大小

        if vocab_size is None:
            vocab_size = len(model_vocab)
        # 如果无法获取词汇表大小，使用词汇表的长度

        p = torch.zeros(len(vocab_out), vocab_size, dtype=torch.int32)
        # 初始化一个零张量，用于存储词汇表的映射

        for tok in model_vocab:
            if tok not in vocab_out:
                continue
            # 对于每个词汇，如果不在最终分词器的词汇表中，则跳过

            orig_idx = model_vocab[tok]
            if orig_idx >= vocab_size:
                logging.warning(
                    f"{model} token {repr(tok)} has index {orig_idx}>{vocab_size-1} (padding?)"
                )
                continue
            # 如果词汇索引超出范围，记录警告并跳过

            new_idx = vocab_out[tok]
            p[new_idx, orig_idx] = 1
        # 更新张量，将原始词汇表的索引映射到最终词汇表的索引

        permutations[model] = p
        # 将映射张量添加到词汇表映射字典中

    return tokenizer_out, permutations
    # 返回最终分词器和词汇表映射字典

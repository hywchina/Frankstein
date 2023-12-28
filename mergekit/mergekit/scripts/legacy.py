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

from typing import List, Optional

import typer
import yaml
from typing_extensions import Annotated

from mergekit.config import InputModelDefinition, MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

# main函数，处理命令行参数并执行模型合并。
def main(
    # out_path: 最终模型的输出目录。
    out_path: Annotated[str, typer.Argument(help="Output directory for final model")],

    # merge: 要合并的模型列表。
    merge: Annotated[
        List[str], typer.Option(help="Add a model to the merge", metavar="MODEL")
    ],

    # density: 每个模型保留权重的比例（仅在 ties 方法中使用）。
    density: Annotated[
        List[float],
        typer.Option(
            help="Fraction of weights to keep for each model (ties only)",
            default_factory=list,
            show_default=False,
        ),
    ],
    # weight: 每个模型的权重（默认为所有模型 1.0，如果未指定）。
    weight: Annotated[
        List[float],
        typer.Option(
            help="Weighting for a model (default 1.0 for all models if not specified)",
            default_factory=list,
            show_default=False,
        ),
    ],
    # method: 用于合并模型的方法。
    method: Annotated[str, typer.Option(help="Method used to merge models")] = "ties",
    # base_model: 合并中的基础模型。
    base_model: Annotated[
        Optional[str], typer.Option(help="Base model for merge")
    ] = None,
    # normalize: 是否将合并的参数除以权重之和。
    normalize: Annotated[
        bool,
        typer.Option(
            help="Divide merged parameters by the sum of weights",
        ),
    ] = True,
    # merged_cache_dir: 合并的LoRA模型的存储路径。
    merged_cache_dir: Annotated[
        Optional[str], typer.Option(help="Storage path for merged LoRA models")
    ] = None,
    # cache_dir: 覆盖下载模型的存储路径。
    cache_dir: Annotated[
        Optional[str], typer.Option(help="Override storage path for downloaded models")
    ] = None,
    # cuda: 是否使用CUDA。
    cuda: bool = False,
    # int8_mask: 是否以int8格式存储中间掩码以节省内存。
    int8_mask: Annotated[
        bool, typer.Option(help="Store intermediate masks in int8 to save memory")
    ] = False,
    # bf16: 是否使用bfloat16。
    bf16: Annotated[bool, typer.Option(help="Use bfloat16")] = True,
    # naive_count: 是否使用简单的符号计数而不是权重（仅在 ties 方法中使用）。
    naive_count: Annotated[
        bool, typer.Option(help="Use naive sign count instead of weight (ties only)")
    ] = False,
    # copy_tokenizer: 是否将基础模型的分词器复制到输出中。
    copy_tokenizer: Annotated[
        bool, typer.Option(help="Copy base model tokenizer into output")
    ] = True,
    # print_yaml: 是否打印生成的YAML配置。
    print_yaml: Annotated[
        bool, typer.Option(help="Print generated YAML configuration")
    ] = False,
    # allow_crimes: 是否允许混合架构。
    allow_crimes: Annotated[
        bool, typer.Option(help="Allow mixing architectures")
    ] = False,
):
    """Wrapper for using a subset of legacy-style script arguments."""
    # 使用传统脚本参数的包装函数。
    models = [InputModelDefinition(model=model, parameters={}) for model in merge]
    if base_model and base_model not in merge:
        models.append(InputModelDefinition(model=base_model, parameters={}))

    parameters = {}

    if density:
        if len(density) == 1:
            density = [density[0]] * len(models)
        for idx, d in enumerate(density):
            models[idx].parameters["density"] = d

    if method == "slerp":
        assert len(weight) == 1, "Must specify exactly one weight for SLERP"
        parameters["t"] = weight[0]
    else:
        if weight:
            if len(weight) == 1:
                weight = [weight[0]] * len(models)
            for idx, w in enumerate(weight):
                models[idx].parameters["weight"] = w

    if int8_mask:
        parameters["int8_mask"] = True
    if naive_count:
        parameters["consensus_method"] = "count"
    parameters["normalize"] = normalize

    merge_config = MergeConfiguration(
        merge_method=method,
        models=models,
        parameters=parameters,
        base_model=base_model,
        dtype="bfloat16" if bf16 else None,
    )

    if print_yaml:
        print(yaml.dump(merge_config.model_dump(mode="json", exclude_none=True)))

    run_merge(
        merge_config,
        out_path,
        options=MergeOptions(
            lora_merge_cache=merged_cache_dir,
            transformers_cache=cache_dir,
            cuda=cuda,
            copy_tokenizer=copy_tokenizer,
            allow_crimes=allow_crimes,
        ),
    )


def _main():
    typer.run(main)


if __name__ == "__main__":
    _main()

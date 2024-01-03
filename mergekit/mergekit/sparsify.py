from enum import Enum

import torch
"""
这段代码提供了对 PyTorch 张量进行稀疏化处理的功能，包括几种不同的稀疏化方法。
以下是对这段代码的逐行中文注释及其逻辑关系的解释：

"""

class SparsificationMethod(str, Enum):
    # 定义一个枚举类 SparsificationMethod，表示不同的稀疏化方法。
    # 枚举类包含三种方法：magnitude（基于大小）、random（随机）和 rescaled_random（缩放后的随机）。
    magnitude = "magnitude"
    random = "random"
    rescaled_random = "rescaled_random"


def magnitude(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """Masks out the smallest values, retaining a proportion of `density`."""
    # 定义一个函数 magnitude，用于保留张量中最大的一部分值，数量由 density 决定。
    # 如果密度大于或等于1，则返回原始张量。
    if density >= 1:
        return tensor
    
    # 计算应保留的元素数量。
    k = int(density * tensor.view(-1).shape[0])

    # 确保至少保留一个元素，避免将整个张量置零。    
    assert k > 0, "not gonna zero out the whole tensor buddy"
    mask = torch.zeros_like(tensor)
    w = tensor.abs().view(-1)
    if w.device.type == "cpu":
        w = w.float()
    topk = torch.topk(w, k=k, largest=True)

    # 创建一个掩码，将张量中最大的 k 个元素保留下来。
    mask.view(-1)[topk.indices] = 1

    # 返回应用掩码后的张量。
    return tensor * mask

# 定义一个函数 bernoulli，用于随机掩码张量值，可选择是否缩放。
def bernoulli(
    tensor: torch.Tensor, density: float, rescale: bool = True
) -> torch.Tensor:
    
    # 如果密度大于或等于1，则返回原始张量。
    if density >= 1:
        return tensor

     # 创建一个伯努利掩码，并应用于张量。
    mask = torch.bernoulli(torch.full_like(input=tensor, fill_value=density))
    res = tensor * mask

    # 如果启用缩放，则对结果张量进行缩放。
    if rescale:
        res /= density
    return res

# 定义一个函数 sparsify，应用稀疏化方法到张量。
def sparsify(
    tensor: torch.Tensor, density: float, method: SparsificationMethod
) -> torch.Tensor:
    
    # 根据指定的方法应用稀疏化处理。
    if method == SparsificationMethod.magnitude:
        return magnitude(tensor, density=density)
    elif method == SparsificationMethod.random:
        return bernoulli(tensor, density=density, rescale=False)
    elif method == SparsificationMethod.rescaled_random:
        return bernoulli(tensor, density=density, rescale=True)
    else:
        raise NotImplementedError(method)

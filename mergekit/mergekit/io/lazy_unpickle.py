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

import codecs
import collections
import contextlib
import operator
import os
import pickle
import zipfile
from functools import reduce
from typing import Any, Optional, Tuple, Union

import accelerate
import numpy
import torch
from pydantic import BaseModel, PrivateAttr

ACCEPTABLE_TYPES = {
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("numpy.core.multiarray", "scalar"): numpy.core.multiarray.scalar,
    ("numpy", "dtype"): numpy.core.multiarray.scalar,
    ("_codecs", "encode"): codecs.encode,
    **{
        ("torch", name): getattr(torch, name)
        for name in [
            "DoubleStorage",
            "FloatStorage",
            "HalfStorage",
            "LongStorage",
            "IntStorage",
            "ShortStorage",
            "CharStorage",
            "ByteStorage",
            "BoolStorage",
            "BFloat16Storage",
        ]
    },
}


class DeferredLoad(BaseModel, arbitrary_types_allowed=True):
    # 定义一个名为 DeferredLoad 的类，允许任意类型的属性

    name: str
    location: str
    dtype: torch.dtype
    # 类的基本属性，包括名称、位置和数据类型

    # 在构造后由 rebuild() 设置
    file_offset: Optional[int] = None
    shape: Optional[Union[torch.Size, Tuple[int, ...]]] = None
    stride: Optional[Tuple[int, ...]] = None
    # 可选属性，包括文件偏移量、形状和步长

    # 在 PyTorch 内部任意设置
    requires_grad: bool = False
    _backward_hooks: Any = PrivateAttr(None)
    # 属性，指定是否需要梯度和私有属性 _backward_hooks


    @staticmethod
    def rebuild(
        load: "DeferredLoad",
        offset: int,
        shape: Union[torch.Size, Tuple[int, ...]],
        stride: Tuple[int, ...],
    ) -> "DeferredLoad":
        # 静态方法，用于重构 DeferredLoad 对象

        load.shape = shape
        load.stride = stride
        load.file_offset = offset * dtype_bytes(load.dtype)
        # 设置形状、步长和文件偏移量

        return load
        # 返回重构后的 DeferredLoad 对象


    def execute(
        self,
        reader: "TorchArchiveReader",
        map_location: Any = None,
    ) -> torch.Tensor:
        # 方法，用于执行延迟加载并返回张量

        total_params = reduce(operator.mul, self.shape)
        total_bytes = total_params * dtype_bytes(self.dtype)
        # 计算总参数数量和字节大小

        f = reader.open_file(file_name=self.name, offset=self.file_offset)
        # 打开文件

        storage = torch.UntypedStorage.from_buffer(
            f.read(total_bytes), "little", dtype=self.dtype
        )
        # 从缓冲区创建未类型化存储

        storage = torch.serialization._get_restore_location(map_location)(
            storage, self.location
        )
        # 获取还原位置并应用于存储

        tensor = torch.tensor([], dtype=self.dtype, device=storage.device)
        tensor.set_(storage, 0, self.shape, self.stride)
        # 创建张量并设置存储、形状和步长

        tensor.requires_grad = self.requires_grad
        tensor._backward_hooks = self._backward_hooks
        # 设置是否需要梯度和 _backward_hooks

        return tensor
        # 返回张量



class LazyTorchUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        # 重写 find_class 方法，用于在反序列化时找到特定的类或函数

        if (module, name) in ACCEPTABLE_TYPES:
            return ACCEPTABLE_TYPES[(module, name)]
            # 如果 (module, name) 在 ACCEPTABLE_TYPES 字典中，返回相应的类或函数

        raise pickle.UnpicklingError(f"Unsupported type {module}.{name}")
        # 如果 (module, name) 不在字典中，抛出反序列化错误


    def persistent_load(self, pid: Any) -> Any:
        # 重写 persistent_load 方法，用于处理持久化对象的加载

        if not isinstance(pid, tuple) or pid[0] != "storage":
            raise RuntimeError(f"Unpickling object with unexpected PID: {repr(pid)}")
            # 如果 pid 不是元组或第一个元素不是 "storage"，抛出运行时错误

        storage_type, key, location, _ = pid[1:]
        # 从 pid 中解析存储类型、键、位置等信息

        return DeferredLoad(name=key, location=location, dtype=get_dtype(storage_type))
        # 返回一个 DeferredLoad 对象，用于延迟加载存储


class TorchArchiveReader:
    """
    用于从 torch ZIP 存档中懒加载文件或文件部分的类。

    维护对最近打开的文件的句柄，以便在从同一文件连续读取时更快地访问。
    """

    archive: zipfile.ZipFile
    archive_name: str
    file_name: Optional[str] = None
    file: Optional[zipfile.ZipExtFile] = None
    # 类的属性，包括 ZIP 存档、存档名称、文件名称和文件句柄


    def __init__(self, path: str):
        self.archive = zipfile.ZipFile(path, mode="r")
        # 打开 ZIP 存档

        self.archive_name = os.path.basename(os.path.normpath(path)).split(".")[0]
        # 获取存档名称

    def open_file(self, file_name: str, offset: int = 0) -> zipfile.ZipExtFile:
        # 定义一个方法，用于打开存档中的文件

        if self.file_name != file_name or (
            self.file is not None and self.file.tell() > offset
        ):
            # 如果当前文件不是目标文件，或文件已经打开且读取位置大于指定偏移量

            if self.file is not None:
                self.file.close()
                # 关闭当前打开的文件

            try:
                fd = self.archive.open(f"archive/data/{file_name}", mode="r")
            except Exception:
                fd = self.archive.open(
                    f"{self.archive_name}/data/{file_name}", mode="r"
                )
            # 尝试打开目标文件，如果失败则尝试另一种路径格式

            self.file = fd
            self.file_name = file_name
            # 更新文件句柄和文件名

        skip_bytes = offset - self.file.tell()
        assert skip_bytes >= 0
        self.file.seek(skip_bytes, os.SEEK_CUR)
        # 移动文件指针到指定偏移量

        return self.file
        # 返回文件句柄



@contextlib.contextmanager
def torch_lazy_load():
    """
    上下文管理器，在此作用域内 `torch.load` 将返回 `DeferredLoad` 而非 `torch.Tensor`。
    """
    old_unpickler = pickle.Unpickler
    old_load = pickle.load
    old_rebuild_tensor = torch._utils._rebuild_tensor
    # 保存原始的 unpickler、load 函数和 rebuild_tensor 函数

    try:
        def load_monkeypatch(*args, **kwargs):
            return pickle.Unpickler(*args, **kwargs).load()
        # 定义一个补丁函数，用于替换 pickle.load

        pickle.Unpickler = LazyTorchUnpickler
        pickle.load = load_monkeypatch
        torch._utils._rebuild_tensor = DeferredLoad.rebuild
        # 替换原始的 unpickler、load 函数和 rebuild_tensor 函数

        with accelerate.init_empty_weights():
            yield
        # 在修改后的环境中执行代码块

    finally:
        torch._utils._rebuild_tensor = old_rebuild_tensor
        pickle.Unpickler = old_unpickler
        pickle.load = old_load
        # 在退出上下文时恢复原始的 unpickler、load 函数和 rebuild_tensor 函数


def dtype_bytes(dtype: torch.dtype) -> int:
    """返回存储 `dtype` 类型单个实例所需的字节数。"""
    if dtype.is_floating_point:
        ti = torch.finfo(dtype)
    else:
        ti = torch.iinfo(dtype)
    # 根据数据类型是浮点数还是整数，选择相应的信息类

    return max(1, ti.bits // 8)
    # 返回每个实例所需的字节数



def get_dtype(storage_type: Any):
    # 获取给定存储类型的 dtype

    if isinstance(storage_type, torch.dtype):
        return storage_type
    # 如果 storage_type 已经是 torch.dtype 类型，直接返回

    dtype = storage_type.dtype
    if not isinstance(dtype, torch.dtype):
        dtype = storage_type(0).dtype
    # 如果 storage_type 的 dtype 属性不是 torch.dtype 类型，创建一个实例并获取其 dtype

    return dtype
    # 返回 dtype

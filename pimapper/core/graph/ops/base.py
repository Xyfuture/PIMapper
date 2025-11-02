"""Op 基类定义

定义计算图中所有操作的基础接口。所有 Op 都应该继承自这个基类。
"""

from __future__ import annotations

from abc import ABC
from collections import OrderedDict
from typing import Any, ClassVar, Optional


class GraphTensor:
    """记录 Tensor 的维度信息，不存储实际值"""

    def __init__(self, shape: Optional[tuple[int, ...]] = None, dtype: Optional[str] = None):
        self.shape = shape
        self.dtype = dtype


class GraphTensorSlice:
    """记录对 GraphTensor 的 slice 操作信息"""

    def __init__(self, slices: tuple[Any, ...]):
        self.slices = slices


class ResultFilter:
    """对 Op 的 results 进行筛选"""

    def __init__(self, index: int = 0, tensor_slice: Optional[GraphTensorSlice] = None):
        self.index = index
        self.tensor_slice = tensor_slice


class Op(ABC):
    """Op 基类

    定义计算图中所有操作的基础接口。每个 Op 实例代表计算图中的一个节点。
    """

    op_type: ClassVar[str] = "base_op"

    def __init__(self, args: Optional[tuple[Any, ...]] = None, kwargs: Optional[dict[str, Any]] = None):
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.input_ops: OrderedDict[Op, ResultFilter] = OrderedDict()
        self.results: list[GraphTensor] = []

    @classmethod
    def convert_from_torch(cls, torch_op: Any) -> Op:
        """从 TorchFxOp 转换，子类实现"""
        raise NotImplementedError(f"{cls.__name__} does not implement convert_from_torch")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

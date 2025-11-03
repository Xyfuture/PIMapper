"""Op 基类定义

定义计算图中所有操作的基础接口。所有 Op 都应该继承自这个基类。
"""

from __future__ import annotations

from abc import ABC
from collections import OrderedDict
from typing import Any, ClassVar, Optional


class GraphTensor:
    """记录 Tensor 的维度信息，不存储实际值

    Attributes:
        shape: Tensor 的形状，例如 (batch, seq_len, hidden_dim)
        dtype: Tensor 的数据类型，例如 "torch.float32"
    """

    def __init__(self, shape: Optional[tuple[int, ...]] = None, dtype: Optional[str] = None):
        self.shape = shape
        self.dtype = dtype


class TensorTransform:
    """记录对 GraphTensor 的变换操作

    支持多种 tensor 变换操作，包括 slice, transpose, view, reshape, permute 等。
    每个 TensorTransform 实例记录一个变换操作及其参数。

    支持的操作类型：
    - slice: 切片操作，参数为 slices (tuple of slice objects or indices)
    - transpose: 转置操作，参数为 dim0, dim1 (两个维度索引)
    - permute: 维度重排，参数为 dims (维度顺序的 tuple)
    - view: 视图变换，参数为 shape (新的形状 tuple)
    - reshape: 形状重塑，参数为 shape (新的形状 tuple)
    - squeeze: 压缩维度，参数为 dim (可选，要压缩的维度索引)
    - unsqueeze: 扩展维度，参数为 dim (要扩展的维度索引)
    - flatten: 展平操作，参数为 start_dim, end_dim (起始和结束维度)
    - expand: 扩展操作，参数为 sizes (扩展后的形状)
    - contiguous: 连续化操作，无参数
    - gather_blocks: 块收集操作，参数为 blocks (块列表), dim (拼接维度)
    - strided_slice: 步长切片操作，参数为 start, end, stride, block_size, dim

    Attributes:
        op_type: 操作类型字符串
        params: 操作参数字典
    """

    def __init__(self, op_type: str, **params: Any):
        """初始化 TensorTransform

        Args:
            op_type: 操作类型，如 "slice", "transpose", "view" 等
            **params: 操作参数，根据不同操作类型有不同的参数
        """
        self.op_type = op_type
        self.params = params

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"TensorTransform({self.op_type}, {params_str})"

    @classmethod
    def slice(cls, slices: tuple[Any, ...]) -> TensorTransform:
        """创建切片操作

        Args:
            slices: 切片参数，例如 (slice(0, 10), slice(None), 0)
        """
        return cls("slice", slices=slices)

    @classmethod
    def transpose(cls, dim0: int, dim1: int) -> TensorTransform:
        """创建转置操作

        Args:
            dim0: 第一个维度索引
            dim1: 第二个维度索引
        """
        return cls("transpose", dim0=dim0, dim1=dim1)

    @classmethod
    def permute(cls, dims: tuple[int, ...]) -> TensorTransform:
        """创建维度重排操作

        Args:
            dims: 新的维度顺序，例如 (0, 2, 1) 表示交换第1和第2维
        """
        return cls("permute", dims=dims)

    @classmethod
    def view(cls, shape: tuple[int, ...]) -> TensorTransform:
        """创建视图变换操作

        Args:
            shape: 新的形状，例如 (batch, -1, hidden_dim)
        """
        return cls("view", shape=shape)

    @classmethod
    def reshape(cls, shape: tuple[int, ...]) -> TensorTransform:
        """创建形状重塑操作

        Args:
            shape: 新的形状，例如 (batch, -1, hidden_dim)
        """
        return cls("reshape", shape=shape)

    @classmethod
    def squeeze(cls, dim: Optional[int] = None) -> TensorTransform:
        """创建压缩维度操作

        Args:
            dim: 要压缩的维度索引，None 表示压缩所有大小为1的维度
        """
        return cls("squeeze", dim=dim)

    @classmethod
    def unsqueeze(cls, dim: int) -> TensorTransform:
        """创建扩展维度操作

        Args:
            dim: 要扩展的维度索引
        """
        return cls("unsqueeze", dim=dim)

    @classmethod
    def flatten(cls, start_dim: int = 0, end_dim: int = -1) -> TensorTransform:
        """创建展平操作

        Args:
            start_dim: 起始维度
            end_dim: 结束维度
        """
        return cls("flatten", start_dim=start_dim, end_dim=end_dim)

    @classmethod
    def expand(cls, sizes: tuple[int, ...]) -> TensorTransform:
        """创建扩展操作

        Args:
            sizes: 扩展后的形状
        """
        return cls("expand", sizes=sizes)

    @classmethod
    def contiguous(cls) -> TensorTransform:
        """创建连续化操作"""
        return cls("contiguous")

    @classmethod
    def gather_blocks(cls, blocks: list[tuple[int, int]], dim: int = -1) -> TensorTransform:
        """创建块收集操作

        从多个不连续的位置提取数据块并拼接。

        Args:
            blocks: 块列表，每个元素是 (offset, size) 元组
            dim: 拼接的维度

        Returns:
            TensorTransform 对象
        """
        return cls("gather_blocks", blocks=blocks, dim=dim)

    @classmethod
    def strided_slice(cls, start: int, end: int, stride: int, block_size: int, dim: int = -1) -> TensorTransform:
        """创建步长切片操作

        用于交错模式下的数据提取，每隔 stride 个元素提取 block_size 个元素。

        Args:
            start: 起始位置
            end: 结束位置
            stride: 步长（每隔多少个元素）
            block_size: 每次提取的块大小
            dim: 操作的维度

        Returns:
            TensorTransform 对象

        Example:
            strided_slice(0, 100, 20, 10, dim=-1)
            表示从位置 0 开始，每隔 20 个元素提取 10 个元素，直到位置 100
            提取的位置：[0:10], [20:30], [40:50], [60:70], [80:90]
        """
        return cls("strided_slice", start=start, end=end, stride=stride, block_size=block_size, dim=dim)


class ResultFilter:
    """对 Op 的 results 进行筛选和变换

    支持选择特定的输出索引，并对其应用一系列 TensorTransform 操作。
    多个变换操作会按顺序级联执行。

    Attributes:
        index: 选择的输出索引（Op 可能有多个输出）
        transforms: TensorTransform 操作列表，按顺序执行

    Example:
        # 选择第0个输出，然后进行转置和切片
        filter = ResultFilter(index=0)
        filter.add_transform(TensorTransform.transpose(0, 1))
        filter.add_transform(TensorTransform.slice((slice(0, 10), slice(None))))
    """

    def __init__(self, index: int = 0, transforms: Optional[list[TensorTransform]] = None):
        """初始化 ResultFilter

        Args:
            index: 选择的输出索引，默认为 0
            transforms: TensorTransform 操作列表，默认为空列表
        """
        self.index = index
        self.transforms = transforms if transforms is not None else []

    def add_transform(self, transform: TensorTransform) -> ResultFilter:
        """添加一个变换操作到变换链

        Args:
            transform: 要添加的 TensorTransform 操作

        Returns:
            self，支持链式调用
        """
        self.transforms.append(transform)
        return self

    def clear_transforms(self) -> ResultFilter:
        """清空所有变换操作

        Returns:
            self，支持链式调用
        """
        self.transforms.clear()
        return self

    def __repr__(self) -> str:
        if not self.transforms:
            return f"ResultFilter(index={self.index})"
        transforms_str = ", ".join(str(t) for t in self.transforms)
        return f"ResultFilter(index={self.index}, transforms=[{transforms_str}])"


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

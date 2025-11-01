"""Op 基类定义

定义计算图中所有操作的基础接口。所有 Op 都应该继承自这个基类。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional


@dataclass
class OpMetadata:
    """Op 元数据

    存储操作的额外信息，如形状、数据类型、性能特征等。

    Attributes:
        shape: 输出张量形状（可选）
        dtype: 输出数据类型（可选）
        compute_intensity: 计算强度（FLOP/Byte）
        memory_footprint: 内存占用（字节）
        custom: 自定义元数据字典
    """

    shape: Optional[tuple[int, ...]] = None
    dtype: Optional[str] = None
    compute_intensity: Optional[float] = None
    memory_footprint: Optional[int] = None
    custom: dict[str, Any] = field(default_factory=dict)


class Op(ABC):
    """Op 基类

    定义计算图中所有操作的基础接口。每个 Op 实例代表计算图中的一个节点。

    主要设计理念：
    1. 每个 Op 类代表一种操作类型（如 MatMul, VectorAdd）
    2. Op 实例存储该操作的具体参数和元数据
    3. Op 类可以注册到 normalize pass 中，用于从 torch op 转换

    Attributes:
        op_type: 操作类型的唯一标识符（由子类定义）
        args: 位置参数（通常是输入节点的引用）
        kwargs: 关键字参数（如维度、轴等操作参数）
        metadata: 操作元数据
    """

    # 类级别属性：操作类型标识符（子类必须覆盖）
    op_type: ClassVar[str] = "base_op"

    def __init__(
        self,
        args: Optional[tuple[Any, ...]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        *,
        metadata: Optional[OpMetadata] = None,
    ):
        """初始化 Op

        Args:
            args: 位置参数（可选）
            kwargs: 关键字参数（可选）
            metadata: 操作元数据（可选）
        """
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.metadata = metadata or OpMetadata()

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """将 Op 转换为字典格式（用于序列化）

        Returns:
            包含 op 信息的字典，至少包含：
            - op_type: 操作类型
            - args: 位置参数
            - kwargs: 关键字参数
            - metadata: 元数据（如果有）
        """
        return {
            "op_type": self.op_type,
            "args": self.args,
            "kwargs": self.kwargs,
            "metadata": {
                "shape": self.metadata.shape,
                "dtype": self.metadata.dtype,
                "compute_intensity": self.metadata.compute_intensity,
                "memory_footprint": self.metadata.memory_footprint,
                "custom": self.metadata.custom,
            },
        }

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Op:
        """从字典格式创建 Op（用于反序列化）

        Args:
            data: 包含 op 信息的字典

        Returns:
            Op 实例
        """
        raise NotImplementedError

    @classmethod
    def can_convert_from_torch(cls, op: str, target: Any) -> bool:
        """判断是否可以从 torch op 转换为当前 op

        子类可以覆盖此方法来实现自动转换逻辑。

        Args:
            op: torch 操作类型（如 'call_function', 'call_module'）
            target: torch 目标（如函数名、模块名）

        Returns:
            True 如果可以转换，False 否则
        """
        return False

    @classmethod
    def convert_from_torch(
        cls,
        op: str,
        target: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        metadata: Optional[OpMetadata] = None,
    ) -> Op:
        """从 torch op 转换为当前 op

        子类应该覆盖此方法来实现具体的转换逻辑。

        Args:
            op: torch 操作类型
            target: torch 目标
            args: 原始参数
            kwargs: 原始关键字参数
            metadata: 元数据（可选）

        Returns:
            转换后的 Op 实例

        Raises:
            NotImplementedError: 如果子类未实现此方法
        """
        raise NotImplementedError(f"{cls.__name__} does not implement convert_from_torch")

    def get_input_refs(self) -> list[str]:
        """获取所有输入节点引用

        从 args 和 kwargs 中提取节点名称字符串。

        Returns:
            输入节点名称列表
        """
        refs = []

        def _extract_refs(obj: Any) -> None:
            if isinstance(obj, str):
                refs.append(obj)
            elif isinstance(obj, (tuple, list)):
                for item in obj:
                    _extract_refs(item)
            elif isinstance(obj, dict):
                for value in obj.values():
                    _extract_refs(value)

        _extract_refs(self.args)
        _extract_refs(self.kwargs)
        return refs

    def __repr__(self) -> str:
        args_str = f"{len(self.args)} args" if self.args else "no args"
        kwargs_str = f"{len(self.kwargs)} kwargs" if self.kwargs else "no kwargs"
        return f"{self.__class__.__name__}({args_str}, {kwargs_str})"

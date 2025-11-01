"""Torch.fx 兼容层

实现 torch.fx 中所有操作类型的 Op 封装。
这些操作可以通过 normalize pass 转换为原生操作。
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

from pimapper.core.graph.ops.base import Op, OpMetadata


class TorchFxOp(Op):
    """Torch.fx 操作的基类

    所有 torch.fx 操作类型的通用基类，提供统一的接口。

    Attributes:
        target: torch.fx 目标（函数、方法名、模块名、属性名等）
    """

    op_type: ClassVar[str] = "torch_fx_base"

    def __init__(
        self,
        target: Any = None,
        args: Optional[tuple[Any, ...]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        *,
        metadata: Optional[OpMetadata] = None,
    ):
        """初始化 Torch.fx 操作

        Args:
            target: torch.fx 目标（可选）
            args: 位置参数（可选）
            kwargs: 关键字参数（可选）
            metadata: 元数据（可选）
        """
        super().__init__(args=args, kwargs=kwargs, metadata=metadata)
        self.target = target

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        if self.target is not None:
            data["target"] = str(self.target)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TorchFxOp:
        metadata = OpMetadata(
            shape=data.get("metadata", {}).get("shape"),
            dtype=data.get("metadata", {}).get("dtype"),
            compute_intensity=data.get("metadata", {}).get("compute_intensity"),
            memory_footprint=data.get("metadata", {}).get("memory_footprint"),
            custom=data.get("metadata", {}).get("custom", {}),
        )
        return cls(
            target=data.get("target"),
            args=tuple(data.get("args", ())),
            kwargs=data.get("kwargs", {}),
            metadata=metadata,
        )


class TorchPlaceholderOp(TorchFxOp):
    """占位符操作（placeholder）

    表示计算图的输入节点，对应 torch.fx 的 placeholder 操作。

    Attributes:
        name: 输入参数名称
    """

    op_type: ClassVar[str] = "placeholder"

    def __init__(
        self,
        name: str,
        *,
        metadata: Optional[OpMetadata] = None,
    ):
        """初始化占位符

        Args:
            name: 输入参数名称
            metadata: 元数据（可选）
        """
        super().__init__(target=name, args=(), kwargs={}, metadata=metadata)
        self.name = name

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["name"] = self.name
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TorchPlaceholderOp:
        metadata = OpMetadata(
            shape=data.get("metadata", {}).get("shape"),
            dtype=data.get("metadata", {}).get("dtype"),
            compute_intensity=data.get("metadata", {}).get("compute_intensity"),
            memory_footprint=data.get("metadata", {}).get("memory_footprint"),
            custom=data.get("metadata", {}).get("custom", {}),
        )
        return cls(name=data["name"], metadata=metadata)

    def __repr__(self) -> str:
        return f"TorchPlaceholderOp(name={self.name})"


class TorchCallFunctionOp(TorchFxOp):
    """函数调用操作（call_function）

    表示对 Python 函数的调用，如 torch.add, torch.matmul 等。
    对应 torch.fx 的 call_function 操作。

    Attributes:
        target: 要调用的函数
    """

    op_type: ClassVar[str] = "call_function"

    def __init__(
        self,
        target: Any,
        args: Optional[tuple[Any, ...]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        *,
        metadata: Optional[OpMetadata] = None,
    ):
        """初始化 call_function 操作

        Args:
            target: 要调用的函数
            args: 位置参数（可选）
            kwargs: 关键字参数（可选）
            metadata: 元数据（可选）
        """
        super().__init__(target=target, args=args, kwargs=kwargs, metadata=metadata)

    def __repr__(self) -> str:
        return f"TorchCallFunctionOp(target={self.target})"


class TorchCallMethodOp(TorchFxOp):
    """方法调用操作（call_method）

    表示对对象方法的调用，如 tensor.add, tensor.view 等。
    第一个参数通常是对象本身。
    对应 torch.fx 的 call_method 操作。

    Attributes:
        method_name: 方法名称
    """

    op_type: ClassVar[str] = "call_method"

    def __init__(
        self,
        method_name: str,
        args: Optional[tuple[Any, ...]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        *,
        metadata: Optional[OpMetadata] = None,
    ):
        """初始化 call_method 操作

        Args:
            method_name: 方法名称
            args: 位置参数（可选，第一个参数为对象本身）
            kwargs: 关键字参数（可选）
            metadata: 元数据（可选）
        """
        super().__init__(target=method_name, args=args, kwargs=kwargs, metadata=metadata)
        self.method_name = method_name

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["method_name"] = self.method_name
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TorchCallMethodOp:
        metadata = OpMetadata(
            shape=data.get("metadata", {}).get("shape"),
            dtype=data.get("metadata", {}).get("dtype"),
            compute_intensity=data.get("metadata", {}).get("compute_intensity"),
            memory_footprint=data.get("metadata", {}).get("memory_footprint"),
            custom=data.get("metadata", {}).get("custom", {}),
        )
        return cls(
            method_name=data["method_name"],
            args=tuple(data.get("args", ())),
            kwargs=data.get("kwargs", {}),
            metadata=metadata,
        )

    def __repr__(self) -> str:
        return f"TorchCallMethodOp(method={self.method_name})"


class TorchCallModuleOp(TorchFxOp):
    """模块调用操作（call_module）

    表示对神经网络模块的调用，如 Linear, Attention 等。
    对应 torch.fx 的 call_module 操作。

    Attributes:
        module_name: 模块的完全限定名称
    """

    op_type: ClassVar[str] = "call_module"

    def __init__(
        self,
        module_name: str,
        args: Optional[tuple[Any, ...]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        *,
        metadata: Optional[OpMetadata] = None,
    ):
        """初始化 call_module 操作

        Args:
            module_name: 模块的完全限定名称
            args: 位置参数（可选）
            kwargs: 关键字参数（可选）
            metadata: 元数据（可选）
        """
        super().__init__(target=module_name, args=args, kwargs=kwargs, metadata=metadata)
        self.module_name = module_name

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["module_name"] = self.module_name
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TorchCallModuleOp:
        metadata = OpMetadata(
            shape=data.get("metadata", {}).get("shape"),
            dtype=data.get("metadata", {}).get("dtype"),
            compute_intensity=data.get("metadata", {}).get("compute_intensity"),
            memory_footprint=data.get("metadata", {}).get("memory_footprint"),
            custom=data.get("metadata", {}).get("custom", {}),
        )
        return cls(
            module_name=data["module_name"],
            args=tuple(data.get("args", ())),
            kwargs=data.get("kwargs", {}),
            metadata=metadata,
        )

    def __repr__(self) -> str:
        return f"TorchCallModuleOp(module={self.module_name})"


class TorchGetAttrOp(TorchFxOp):
    """属性获取操作（get_attr）

    表示获取模块属性，如权重、偏置等。
    对应 torch.fx 的 get_attr 操作。

    Attributes:
        attr_name: 属性的完全限定名称
    """

    op_type: ClassVar[str] = "get_attr"

    def __init__(
        self,
        attr_name: str,
        *,
        metadata: Optional[OpMetadata] = None,
    ):
        """初始化 get_attr 操作

        Args:
            attr_name: 属性的完全限定名称
            metadata: 元数据（可选）
        """
        super().__init__(target=attr_name, args=(), kwargs={}, metadata=metadata)
        self.attr_name = attr_name

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["attr_name"] = self.attr_name
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TorchGetAttrOp:
        metadata = OpMetadata(
            shape=data.get("metadata", {}).get("shape"),
            dtype=data.get("metadata", {}).get("dtype"),
            compute_intensity=data.get("metadata", {}).get("compute_intensity"),
            memory_footprint=data.get("metadata", {}).get("memory_footprint"),
            custom=data.get("metadata", {}).get("custom", {}),
        )
        return cls(attr_name=data["attr_name"], metadata=metadata)

    def __repr__(self) -> str:
        return f"TorchGetAttrOp(attr={self.attr_name})"


class TorchOutputOp(TorchFxOp):
    """输出操作（output）

    表示计算图的输出节点，收集最终结果。
    对应 torch.fx 的 output 操作。

    Attributes:
        result: 返回值（单个值或元组）
    """

    op_type: ClassVar[str] = "output"

    def __init__(
        self,
        result: Any,
        *,
        metadata: Optional[OpMetadata] = None,
    ):
        """初始化输出

        Args:
            result: 返回值（单个值或元组）
            metadata: 元数据（可选）
        """
        if isinstance(result, tuple):
            args = result
        else:
            args = (result,)
        super().__init__(target=None, args=args, kwargs={}, metadata=metadata)
        self.result = result

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["result"] = self.result
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TorchOutputOp:
        metadata = OpMetadata(
            shape=data.get("metadata", {}).get("shape"),
            dtype=data.get("metadata", {}).get("dtype"),
            compute_intensity=data.get("metadata", {}).get("compute_intensity"),
            memory_footprint=data.get("metadata", {}).get("memory_footprint"),
            custom=data.get("metadata", {}).get("custom", {}),
        )
        return cls(result=data["result"], metadata=metadata)

    def __repr__(self) -> str:
        return f"TorchOutputOp(result={self.result})"


class TorchRootOp(TorchFxOp):
    """根操作（root）

    表示模块本身的根节点，通常在 torch.fx 图中用于引用模块。
    对应 torch.fx 的 root 操作。
    """

    op_type: ClassVar[str] = "root"

    def __init__(
        self,
        *,
        metadata: Optional[OpMetadata] = None,
    ):
        """初始化根操作

        Args:
            metadata: 元数据（可选）
        """
        super().__init__(target=None, args=(), kwargs={}, metadata=metadata)

    def __repr__(self) -> str:
        return "TorchRootOp()"


def create_torch_op_from_fx(
    op: str,
    target: Any,
    args: Optional[tuple[Any, ...]] = None,
    kwargs: Optional[dict[str, Any]] = None,
    *,
    metadata: Optional[OpMetadata] = None,
) -> TorchFxOp:
    """从 torch.fx 节点信息创建对应的 Op

    工厂函数，根据操作类型自动选择合适的 Op 类。

    Args:
        op: torch.fx 操作类型
        target: torch.fx 目标
        args: 位置参数（可选）
        kwargs: 关键字参数（可选）
        metadata: 元数据（可选）

    Returns:
        对应的 TorchFxOp 实例

    Raises:
        ValueError: 不支持的操作类型
    """
    if op == "placeholder":
        return TorchPlaceholderOp(str(target), metadata=metadata)
    elif op == "call_function":
        return TorchCallFunctionOp(target, args, kwargs, metadata=metadata)
    elif op == "call_method":
        return TorchCallMethodOp(str(target), args, kwargs, metadata=metadata)
    elif op == "call_module":
        return TorchCallModuleOp(str(target), args, kwargs, metadata=metadata)
    elif op == "get_attr":
        return TorchGetAttrOp(str(target), metadata=metadata)
    elif op == "output":
        # 对于 output，args 通常包含返回值
        result = args[0] if args and len(args) == 1 else args
        return TorchOutputOp(result, metadata=metadata)
    elif op == "root":
        return TorchRootOp(metadata=metadata)
    else:
        raise ValueError(f"Unsupported torch.fx op type: {op}")

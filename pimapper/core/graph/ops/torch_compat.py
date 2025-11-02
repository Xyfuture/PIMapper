"""Torch.fx 兼容层

实现 torch.fx 中所有操作类型的 Op 封装。
这些操作可以通过 normalize pass 转换为原生操作。
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

from pimapper.core.graph.ops.base import Op


class TorchFxOp(Op):
    """Torch.fx 操作的基类

    所有 torch.fx 操作类型的通用基类，提供统一的接口。

    Attributes:
        target: torch.fx 目标（函数、方法名、模块名、属性名等）
    """

    op_type: ClassVar[str] = "torch_fx_base"

    def __init__(self, target: Any = None, args: Optional[tuple[Any, ...]] = None, kwargs: Optional[dict[str, Any]] = None, *, metadata: Optional[dict] = None):
        super().__init__(args=args, kwargs=kwargs)
        self.target = target
        self.metadata = metadata or {}


class TorchPlaceholderOp(TorchFxOp):
    op_type: ClassVar[str] = "placeholder"

    def __init__(self, name: str, *, metadata: Optional[dict] = None):
        super().__init__(target=name, args=(), kwargs={}, metadata=metadata)
        self.name = name


class TorchCallFunctionOp(TorchFxOp):
    op_type: ClassVar[str] = "call_function"


class TorchCallMethodOp(TorchFxOp):
    op_type: ClassVar[str] = "call_method"

    def __init__(self, method_name: str, args: Optional[tuple[Any, ...]] = None, kwargs: Optional[dict[str, Any]] = None, *, metadata: Optional[dict] = None):
        super().__init__(target=method_name, args=args, kwargs=kwargs, metadata=metadata)
        self.method_name = method_name


class TorchCallModuleOp(TorchFxOp):
    op_type: ClassVar[str] = "call_module"

    def __init__(self, module_name: str, args: Optional[tuple[Any, ...]] = None, kwargs: Optional[dict[str, Any]] = None, *, metadata: Optional[dict] = None):
        super().__init__(target=module_name, args=args, kwargs=kwargs, metadata=metadata)
        self.module_name = module_name


class TorchGetAttrOp(TorchFxOp):
    op_type: ClassVar[str] = "get_attr"

    def __init__(self, attr_name: str, *, metadata: Optional[dict] = None):
        super().__init__(target=attr_name, args=(), kwargs={}, metadata=metadata)
        self.attr_name = attr_name


class TorchOutputOp(TorchFxOp):
    op_type: ClassVar[str] = "output"

    def __init__(self, result: Any, *, metadata: Optional[dict] = None):
        args = result if isinstance(result, tuple) else (result,)
        super().__init__(target=None, args=args, kwargs={}, metadata=metadata)
        self.result = result


class TorchRootOp(TorchFxOp):
    op_type: ClassVar[str] = "root"

    def __init__(self, *, metadata: Optional[dict] = None):
        super().__init__(target=None, args=(), kwargs={}, metadata=metadata)


def create_torch_op_from_fx(op: str, target: Any, args: Optional[tuple[Any, ...]] = None, kwargs: Optional[dict[str, Any]] = None, *, metadata: Optional[dict] = None) -> TorchFxOp:
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
        result = args[0] if args and len(args) == 1 else args
        return TorchOutputOp(result, metadata=metadata)
    elif op == "root":
        return TorchRootOp(metadata=metadata)
    else:
        raise ValueError(f"Unsupported torch.fx op type: {op}")

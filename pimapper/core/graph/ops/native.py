"""原生操作定义

定义 FusionMachine 的原生操作，这些操作是计算图的核心计算单元。
每个操作都支持从 torch 操作自动转换。
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

from pimapper.core.graph.ops.base import Op, OpMetadata


class MatMulOp(Op):
    """矩阵乘法操作

    执行矩阵乘法 C = A @ B，支持批量矩阵乘法。

    Args in self.args:
        - args[0]: 第一个输入矩阵节点引用（A）
        - args[1]: 第二个输入矩阵节点引用（B）
    """

    op_type: ClassVar[str] = "matmul"

    def __init__(
        self,
        input_a: str,
        input_b: str,
        *,
        transpose_a: bool = False,
        transpose_b: bool = False,
        metadata: Optional[OpMetadata] = None,
    ):
        """初始化矩阵乘法操作

        Args:
            input_a: 第一个输入矩阵节点引用
            input_b: 第二个输入矩阵节点引用
            transpose_a: 是否转置第一个矩阵
            transpose_b: 是否转置第二个矩阵
            metadata: 元数据（可选）
        """
        super().__init__(
            args=(input_a, input_b),
            kwargs={"transpose_a": transpose_a, "transpose_b": transpose_b},
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatMulOp:
        metadata = OpMetadata(
            shape=data.get("metadata", {}).get("shape"),
            dtype=data.get("metadata", {}).get("dtype"),
            custom=data.get("metadata", {}).get("custom", {}),
        )
        return cls(
            input_a=data["args"][0],
            input_b=data["args"][1],
            transpose_a=data["kwargs"].get("transpose_a", False),
            transpose_b=data["kwargs"].get("transpose_b", False),
            metadata=metadata,
        )

    @classmethod
    def can_convert_from_torch(cls, op: str, target: Any, metadata: Optional[OpMetadata] = None) -> bool:
        if op == "call_function":
            target_str = str(target).lower()
            return "matmul" in target_str or "bmm" in target_str or target_str.endswith("mm")
        elif op == "call_module":
            # Check if it's a Linear module
            if metadata and metadata.custom:
                module_class = metadata.custom.get("module_class", "")
                return module_class == "Linear"
        return False

    @classmethod
    def convert_from_torch(
        cls,
        op: str,
        target: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        metadata: Optional[OpMetadata] = None,
    ) -> MatMulOp:
        if op == "call_module" and metadata and metadata.custom:
            # For Linear modules, the first argument is the input
            # We need to create a weight node reference
            # The weight is stored in the module, so we use the module name as weight reference
            if len(args) < 1:
                raise ValueError(f"Linear module requires at least 1 argument, got {len(args)}")
            # For Linear: output = input @ weight.T
            # We'll use the module name (target) as the weight reference
            # Note: Linear computes output = input @ weight.T + bias
            # For now, we ignore bias and just do matmul
            return cls(
                input_a=args[0],
                input_b=str(target),  # Use module name as weight reference
                transpose_b=True,  # Linear uses transposed weight
                metadata=metadata,
            )
        else:
            # For call_function matmul operations
            if len(args) < 2:
                raise ValueError(f"MatMul requires at least 2 arguments, got {len(args)}")
            return cls(
                input_a=args[0],
                input_b=args[1],
                transpose_a=kwargs.get("transpose_a", False),
                transpose_b=kwargs.get("transpose_b", False),
                metadata=metadata,
            )


class VectorAddOp(Op):
    """向量加法操作

    执行逐元素加法 C = A + B，支持广播。

    Args in self.args:
        - args[0]: 第一个输入向量节点引用（A）
        - args[1]: 第二个输入向量节点引用（B）
    """

    op_type: ClassVar[str] = "vector_add"

    def __init__(
        self,
        input_a: str,
        input_b: str,
        *,
        alpha: float = 1.0,
        metadata: Optional[OpMetadata] = None,
    ):
        """初始化向量加法操作

        Args:
            input_a: 第一个输入向量节点引用
            input_b: 第二个输入向量节点引用
            alpha: 第二个输入的缩放因子（C = A + alpha * B）
            metadata: 元数据（可选）
        """
        super().__init__(
            args=(input_a, input_b),
            kwargs={"alpha": alpha},
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VectorAddOp:
        metadata = OpMetadata(
            shape=data.get("metadata", {}).get("shape"),
            dtype=data.get("metadata", {}).get("dtype"),
            custom=data.get("metadata", {}).get("custom", {}),
        )
        return cls(
            input_a=data["args"][0],
            input_b=data["args"][1],
            alpha=data["kwargs"].get("alpha", 1.0),
            metadata=metadata,
        )

    @classmethod
    def can_convert_from_torch(cls, op: str, target: Any) -> bool:
        if op not in ("call_function", "call_method"):
            return False
        target_str = str(target).lower()
        return "add" in target_str and "matmul" not in target_str

    @classmethod
    def convert_from_torch(
        cls,
        op: str,
        target: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        metadata: Optional[OpMetadata] = None,
    ) -> VectorAddOp:
        if len(args) < 2:
            raise ValueError(f"VectorAdd requires at least 2 arguments, got {len(args)}")
        return cls(
            input_a=args[0],
            input_b=args[1],
            alpha=kwargs.get("alpha", 1.0),
            metadata=metadata,
        )


class VectorMulOp(Op):
    """向量乘法操作

    执行逐元素乘法 C = A * B，支持广播。

    Args in self.args:
        - args[0]: 第一个输入向量节点引用（A）
        - args[1]: 第二个输入向量节点引用（B）
    """

    op_type: ClassVar[str] = "vector_mul"

    def __init__(
        self,
        input_a: str,
        input_b: str,
        *,
        metadata: Optional[OpMetadata] = None,
    ):
        """初始化向量乘法操作

        Args:
            input_a: 第一个输入向量节点引用
            input_b: 第二个输入向量节点引用
            metadata: 元数据（可选）
        """
        super().__init__(
            args=(input_a, input_b),
            kwargs={},
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VectorMulOp:
        metadata = OpMetadata(
            shape=data.get("metadata", {}).get("shape"),
            dtype=data.get("metadata", {}).get("dtype"),
            custom=data.get("metadata", {}).get("custom", {}),
        )
        return cls(
            input_a=data["args"][0],
            input_b=data["args"][1],
            metadata=metadata,
        )

    @classmethod
    def can_convert_from_torch(cls, op: str, target: Any) -> bool:
        if op not in ("call_function", "call_method"):
            return False
        target_str = str(target).lower()
        return "mul" in target_str and "matmul" not in target_str

    @classmethod
    def convert_from_torch(
        cls,
        op: str,
        target: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        metadata: Optional[OpMetadata] = None,
    ) -> VectorMulOp:
        if len(args) < 2:
            raise ValueError(f"VectorMul requires at least 2 arguments, got {len(args)}")
        return cls(
            input_a=args[0],
            input_b=args[1],
            metadata=metadata,
        )


class VectorDotOp(Op):
    """向量点积操作

    执行向量点积 c = dot(A, B) = sum(A * B)

    Args in self.args:
        - args[0]: 第一个输入向量节点引用（A）
        - args[1]: 第二个输入向量节点引用（B）
    """

    op_type: ClassVar[str] = "vector_dot"

    def __init__(
        self,
        input_a: str,
        input_b: str,
        *,
        metadata: Optional[OpMetadata] = None,
    ):
        """初始化向量点积操作

        Args:
            input_a: 第一个输入向量节点引用
            input_b: 第二个输入向量节点引用
            metadata: 元数据（可选）
        """
        super().__init__(
            args=(input_a, input_b),
            kwargs={},
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VectorDotOp:
        metadata = OpMetadata(
            shape=data.get("metadata", {}).get("shape"),
            dtype=data.get("metadata", {}).get("dtype"),
            custom=data.get("metadata", {}).get("custom", {}),
        )
        return cls(
            input_a=data["args"][0],
            input_b=data["args"][1],
            metadata=metadata,
        )

    @classmethod
    def can_convert_from_torch(cls, op: str, target: Any) -> bool:
        if op != "call_function":
            return False
        target_str = str(target).lower()
        return "dot" in target_str and "matmul" not in target_str

    @classmethod
    def convert_from_torch(
        cls,
        op: str,
        target: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        metadata: Optional[OpMetadata] = None,
    ) -> VectorDotOp:
        if len(args) < 2:
            raise ValueError(f"VectorDot requires at least 2 arguments, got {len(args)}")
        return cls(
            input_a=args[0],
            input_b=args[1],
            metadata=metadata,
        )


class SiLUOp(Op):
    """SiLU 激活函数操作

    执行 SiLU (Swish) 激活函数 y = x * sigmoid(x)

    Args in self.args:
        - args[0]: 输入节点引用
    """

    op_type: ClassVar[str] = "silu"

    def __init__(
        self,
        input_x: str,
        *,
        metadata: Optional[OpMetadata] = None,
    ):
        """初始化 SiLU 操作

        Args:
            input_x: 输入节点引用
            metadata: 元数据（可选）
        """
        super().__init__(
            args=(input_x,),
            kwargs={},
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SiLUOp:
        metadata = OpMetadata(
            shape=data.get("metadata", {}).get("shape"),
            dtype=data.get("metadata", {}).get("dtype"),
            custom=data.get("metadata", {}).get("custom", {}),
        )
        return cls(
            input_x=data["args"][0],
            metadata=metadata,
        )

    @classmethod
    def can_convert_from_torch(cls, op: str, target: Any) -> bool:
        if op not in ("call_function", "call_method", "call_module"):
            return False
        target_str = str(target).lower()
        return "silu" in target_str or "swish" in target_str

    @classmethod
    def convert_from_torch(
        cls,
        op: str,
        target: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        metadata: Optional[OpMetadata] = None,
    ) -> SiLUOp:
        if len(args) < 1:
            raise ValueError(f"SiLU requires at least 1 argument, got {len(args)}")
        return cls(
            input_x=args[0],
            metadata=metadata,
        )


class SoftmaxOp(Op):
    """Softmax 操作

    执行 Softmax 归一化 y_i = exp(x_i) / sum(exp(x_j))

    Args in self.args:
        - args[0]: 输入节点引用
    """

    op_type: ClassVar[str] = "softmax"

    def __init__(
        self,
        input_x: str,
        *,
        dim: int = -1,
        metadata: Optional[OpMetadata] = None,
    ):
        """初始化 Softmax 操作

        Args:
            input_x: 输入节点引用
            dim: 应用 softmax 的维度
            metadata: 元数据（可选）
        """
        super().__init__(
            args=(input_x,),
            kwargs={"dim": dim},
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SoftmaxOp:
        metadata = OpMetadata(
            shape=data.get("metadata", {}).get("shape"),
            dtype=data.get("metadata", {}).get("dtype"),
            custom=data.get("metadata", {}).get("custom", {}),
        )
        return cls(
            input_x=data["args"][0],
            dim=data["kwargs"].get("dim", -1),
            metadata=metadata,
        )

    @classmethod
    def can_convert_from_torch(cls, op: str, target: Any) -> bool:
        if op not in ("call_function", "call_method", "call_module"):
            return False
        target_str = str(target).lower()
        return "softmax" in target_str

    @classmethod
    def convert_from_torch(
        cls,
        op: str,
        target: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        metadata: Optional[OpMetadata] = None,
    ) -> SoftmaxOp:
        if len(args) < 1:
            raise ValueError(f"Softmax requires at least 1 argument, got {len(args)}")
        return cls(
            input_x=args[0],
            dim=kwargs.get("dim", -1),
            metadata=metadata,
        )


class RMSNormOp(Op):
    """RMS 归一化操作

    执行 RMS 归一化 y = x / sqrt(mean(x^2) + eps) * weight

    Args in self.args:
        - args[0]: 输入节点引用
        - args[1]: 权重节点引用（可选）
    """

    op_type: ClassVar[str] = "rmsnorm"

    def __init__(
        self,
        input_x: str,
        weight: Optional[str] = None,
        *,
        eps: float = 1e-6,
        normalized_shape: Optional[tuple[int, ...]] = None,
        metadata: Optional[OpMetadata] = None,
    ):
        """初始化 RMSNorm 操作

        Args:
            input_x: 输入节点引用
            weight: 权重节点引用（可选）
            eps: 数值稳定性常数
            normalized_shape: 归一化的形状
            metadata: 元数据（可选）
        """
        args = (input_x,) if weight is None else (input_x, weight)
        super().__init__(
            args=args,
            kwargs={"eps": eps, "normalized_shape": normalized_shape},
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RMSNormOp:
        metadata = OpMetadata(
            shape=data.get("metadata", {}).get("shape"),
            dtype=data.get("metadata", {}).get("dtype"),
            custom=data.get("metadata", {}).get("custom", {}),
        )
        weight = data["args"][1] if len(data["args"]) > 1 else None
        return cls(
            input_x=data["args"][0],
            weight=weight,
            eps=data["kwargs"].get("eps", 1e-6),
            normalized_shape=data["kwargs"].get("normalized_shape"),
            metadata=metadata,
        )

    @classmethod
    def can_convert_from_torch(cls, op: str, target: Any, metadata: Optional[OpMetadata] = None) -> bool:
        if op in ("call_function", "call_method"):
            target_str = str(target).lower()
            return "rmsnorm" in target_str or "rms_norm" in target_str
        elif op == "call_module":
            # Check if it's an RMSNorm module
            if metadata and metadata.custom:
                module_class = metadata.custom.get("module_class", "")
                return module_class == "RMSNorm"
        return False

    @classmethod
    def convert_from_torch(
        cls,
        op: str,
        target: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        metadata: Optional[OpMetadata] = None,
    ) -> RMSNormOp:
        if len(args) < 1:
            raise ValueError(f"RMSNorm requires at least 1 argument, got {len(args)}")

        if op == "call_module":
            # For RMSNorm modules, use module name as weight reference
            return cls(
                input_x=args[0],
                weight=str(target),  # Use module name as weight reference
                eps=kwargs.get("eps", 1e-6),
                normalized_shape=kwargs.get("normalized_shape"),
                metadata=metadata,
            )
        else:
            # For call_function/call_method
            weight = args[1] if len(args) > 1 else None
            return cls(
                input_x=args[0],
                weight=weight,
                eps=kwargs.get("eps", 1e-6),
                normalized_shape=kwargs.get("normalized_shape"),
                metadata=metadata,
            )


# 注册所有原生操作
NATIVE_OPS = [
    MatMulOp,
    VectorAddOp,
    VectorMulOp,
    VectorDotOp,
    SiLUOp,
    SoftmaxOp,
    RMSNormOp,
]

"""原生操作定义

定义 FusionMachine 的原生操作，这些操作是计算图的核心计算单元。
每个操作都支持从 torch 操作自动转换。
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

from pimapper.core.graph.ops.base import Op, GraphTensor


class MatMulOp(Op):
    """矩阵乘法操作"""

    op_type: ClassVar[str] = "matmul"

    def __init__(self, transpose_a: bool = False, transpose_b: bool = False, matrix_shape: Optional[tuple[int, int]] = None):
        super().__init__(kwargs={"transpose_a": transpose_a, "transpose_b": transpose_b, "matrix_shape": matrix_shape})

    @classmethod
    def convert_from_torch(cls, torch_op: Any) -> MatMulOp:
        from pimapper.core.graph.ops.torch_compat import TorchCallFunctionOp, TorchCallModuleOp
        if isinstance(torch_op, TorchCallFunctionOp):
            target_str = str(torch_op.target).lower()
            if "matmul" in target_str or "bmm" in target_str or target_str.endswith("mm"):
                return cls()
        elif isinstance(torch_op, TorchCallModuleOp):
            if torch_op.metadata and torch_op.metadata.get("custom", {}).get("module_class") == "Linear":
                return cls(transpose_b=True)
        raise ValueError(f"Cannot convert {torch_op} to MatMulOp")


class VectorAddOp(Op):
    """向量加法操作"""

    op_type: ClassVar[str] = "vector_add"

    def __init__(self, alpha: float = 1.0):
        super().__init__(kwargs={"alpha": alpha})

    @classmethod
    def convert_from_torch(cls, torch_op: Any) -> VectorAddOp:
        from pimapper.core.graph.ops.torch_compat import TorchCallFunctionOp, TorchCallMethodOp
        if isinstance(torch_op, (TorchCallFunctionOp, TorchCallMethodOp)):
            target_str = str(torch_op.target).lower()
            if "add" in target_str and "matmul" not in target_str:
                return cls()
        raise ValueError(f"Cannot convert {torch_op} to VectorAddOp")


class VectorMulOp(Op):
    """向量乘法操作"""

    op_type: ClassVar[str] = "vector_mul"

    def __init__(self):
        super().__init__()

    @classmethod
    def convert_from_torch(cls, torch_op: Any) -> VectorMulOp:
        from pimapper.core.graph.ops.torch_compat import TorchCallFunctionOp, TorchCallMethodOp
        if isinstance(torch_op, (TorchCallFunctionOp, TorchCallMethodOp)):
            target_str = str(torch_op.target).lower()
            if "mul" in target_str and "matmul" not in target_str:
                return cls()
        raise ValueError(f"Cannot convert {torch_op} to VectorMulOp")


class VectorDotOp(Op):
    """向量点积操作"""

    op_type: ClassVar[str] = "vector_dot"

    def __init__(self):
        super().__init__()

    @classmethod
    def convert_from_torch(cls, torch_op: Any) -> VectorDotOp:
        from pimapper.core.graph.ops.torch_compat import TorchCallFunctionOp
        if isinstance(torch_op, TorchCallFunctionOp):
            target_str = str(torch_op.target).lower()
            if "dot" in target_str and "matmul" not in target_str:
                return cls()
        raise ValueError(f"Cannot convert {torch_op} to VectorDotOp")


class SiLUOp(Op):
    """SiLU 激活函数操作"""

    op_type: ClassVar[str] = "silu"

    def __init__(self):
        super().__init__()

    @classmethod
    def convert_from_torch(cls, torch_op: Any) -> SiLUOp:
        from pimapper.core.graph.ops.torch_compat import TorchCallFunctionOp, TorchCallMethodOp, TorchCallModuleOp
        if isinstance(torch_op, (TorchCallFunctionOp, TorchCallMethodOp, TorchCallModuleOp)):
            target_str = str(torch_op.target).lower()
            if "silu" in target_str or "swish" in target_str:
                return cls()
        raise ValueError(f"Cannot convert {torch_op} to SiLUOp")


class SoftmaxOp(Op):
    """Softmax 操作"""

    op_type: ClassVar[str] = "softmax"

    def __init__(self, dim: int = -1):
        super().__init__(kwargs={"dim": dim})

    @classmethod
    def convert_from_torch(cls, torch_op: Any) -> SoftmaxOp:
        from pimapper.core.graph.ops.torch_compat import TorchCallFunctionOp, TorchCallMethodOp, TorchCallModuleOp
        if isinstance(torch_op, (TorchCallFunctionOp, TorchCallMethodOp, TorchCallModuleOp)):
            target_str = str(torch_op.target).lower()
            if "softmax" in target_str:
                return cls()
        raise ValueError(f"Cannot convert {torch_op} to SoftmaxOp")


class RMSNormOp(Op):
    """RMS 归一化操作"""

    op_type: ClassVar[str] = "rmsnorm"

    def __init__(self, eps: float = 1e-6, normalized_shape: Optional[tuple[int, ...]] = None):
        super().__init__(kwargs={"eps": eps, "normalized_shape": normalized_shape})

    @classmethod
    def convert_from_torch(cls, torch_op: Any) -> RMSNormOp:
        from pimapper.core.graph.ops.torch_compat import TorchCallFunctionOp, TorchCallMethodOp, TorchCallModuleOp
        if isinstance(torch_op, (TorchCallFunctionOp, TorchCallMethodOp)):
            target_str = str(torch_op.target).lower()
            if "rmsnorm" in target_str or "rms_norm" in target_str:
                return cls()
        elif isinstance(torch_op, TorchCallModuleOp):
            if torch_op.metadata and torch_op.metadata.get("custom", {}).get("module_class") == "RMSNorm":
                return cls()
        raise ValueError(f"Cannot convert {torch_op} to RMSNormOp")


class BatchedMatMulOp(Op):
    """批量矩阵乘法操作 (with past KV cache)"""

    op_type: ClassVar[str] = "batched_matmul"

    def __init__(
        self,
        batch_size: int = 1,
        num_matmuls: int = 1,
        matmul_shape: Optional[tuple[int, int, int]] = None,  # (M, rows, cols)
        is_qk_matmul: bool = True,
        model_config: Optional[dict] = None,
        inference_config: Optional[dict] = None
    ):
        """
        Args:
            batch_size: Batch size for the operation
            num_matmuls: Number of parallel matmuls (typically num_attention_heads)
            matmul_shape: Shape of each matmul as (M, rows, cols) where M is the sequence dimension
            is_qk_matmul: True for Q@K^T, False for scores@V
            model_config: Dictionary containing model configuration
            inference_config: Dictionary containing inference configuration
        """
        super().__init__(kwargs={
            "batch_size": batch_size,
            "num_matmuls": num_matmuls,
            "matmul_shape": matmul_shape,
            "is_qk_matmul": is_qk_matmul,
            "model_config": model_config,
            "inference_config": inference_config
        })

    @classmethod
    def convert_from_torch(cls, torch_op: Any) -> BatchedMatMulOp:
        from pimapper.core.graph.ops.torch_compat import TorchCallModuleOp
        if isinstance(torch_op, TorchCallModuleOp):
            if torch_op.metadata and torch_op.metadata.get("custom", {}).get("module_class") == "BatchedMatMulWithPast":
                custom_meta = torch_op.metadata.get("custom", {})

                batch_size = custom_meta.get("batch_size", 1)
                num_matmuls = custom_meta.get("num_matmuls", 1)
                matmul_shape = custom_meta.get("matmul_shape")
                is_qk_matmul = custom_meta.get("is_qk_matmul", True)
                model_config = custom_meta.get("model_config")
                inference_config = custom_meta.get("inference_config")

                return cls(
                    batch_size=batch_size,
                    num_matmuls=num_matmuls,
                    matmul_shape=matmul_shape,
                    is_qk_matmul=is_qk_matmul,
                    model_config=model_config,
                    inference_config=inference_config
                )
        raise ValueError(f"Cannot convert {torch_op} to BatchedMatMulOp")


# 注册所有原生操作
NATIVE_OPS = [
    MatMulOp,
    VectorAddOp,
    VectorMulOp,
    VectorDotOp,
    SiLUOp,
    SoftmaxOp,
    RMSNormOp,
    BatchedMatMulOp,
]

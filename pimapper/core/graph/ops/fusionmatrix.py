"""FusionMatrix 操作

用于表示融合后的矩阵运算，记录多个矩阵的融合情况和执行顺序。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

from pimapper.core.graph.ops.base import Op


@dataclass
class MatrixFusionInfo:
    """单个矩阵在融合操作中的信息

    Attributes:
        node_name: 原始计算图节点名称
        target: 原始目标函数或操作
        args: 原始参数
        kwargs: 原始关键字参数
        shape: 输出张量形状
        dtype: 输出数据类型
        latest_start_time: 最晚开工时间（用于排序）
        order_index: 在融合序列中的顺序索引
    """

    node_name: str
    target: Any
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    latest_start_time: float = 0.0
    order_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式

        Returns:
            包含融合信息的字典
        """
        return {
            "node_name": self.node_name,
            "target": str(self.target),
            "args": self.args,
            "kwargs": self.kwargs,
            "shape": self.shape,
            "dtype": self.dtype,
            "latest_start_time": self.latest_start_time,
            "order_index": self.order_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatrixFusionInfo:
        """从字典格式创建实例

        Args:
            data: 包含融合信息的字典

        Returns:
            MatrixFusionInfo 实例
        """
        return cls(
            node_name=data["node_name"],
            target=data["target"],
            args=tuple(data.get("args", ())),
            kwargs=data.get("kwargs", {}),
            shape=tuple(data["shape"]) if data.get("shape") else None,
            dtype=data.get("dtype"),
            latest_start_time=data.get("latest_start_time", 0.0),
            order_index=data.get("order_index", 0),
        )


class FusionMatrixOp(Op):
    """融合矩阵操作

    将多个输入相同的矩阵运算融合为一个操作，按最晚开工时间排序。
    支持交错执行策略来优化内存访问模式。

    Attributes:
        shared_inputs: 共享的输入节点名称列表
        fused_matrices: 融合的矩阵信息列表，按执行顺序排列
        fusion_strategy: 融合策略 ('sequential' 或 'interleaved')
    """

    op_type: ClassVar[str] = "fusion_matrix"

    def __init__(self, shared_inputs: list[str], fusion_strategy: str = "sequential"):
        super().__init__(args=tuple(shared_inputs), kwargs={"fusion_strategy": fusion_strategy})
        self.shared_inputs = shared_inputs
        self.fused_matrices: list[MatrixFusionInfo] = []
        self.fusion_strategy = fusion_strategy

    def add_matrix(
        self,
        node_name: str,
        target: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        latest_start_time: float,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: str | None = None,
    ) -> None:
        """添加一个矩阵到融合操作

        Args:
            node_name: 原始节点名称
            target: 原始目标函数/操作
            args: 原始参数
            kwargs: 原始关键字参数
            latest_start_time: 最晚开工时间
            shape: 输出形状（可选）
            dtype: 输出数据类型（可选）
        """
        matrix_info = MatrixFusionInfo(
            node_name=node_name,
            target=target,
            args=args,
            kwargs=kwargs,
            shape=shape,
            dtype=dtype,
            latest_start_time=latest_start_time,
            order_index=len(self.fused_matrices),
        )
        self.fused_matrices.append(matrix_info)

    def sort_matrices(self) -> None:
        """根据最晚开工时间和融合策略对矩阵排序

        - sequential: 简单按最晚开工时间升序排序
        - interleaved: 按最晚开工时间分组，组内交错排列
        """
        if self.fusion_strategy == "sequential":
            # 按最晚开工时间升序排序，时间早的先执行
            self.fused_matrices.sort(key=lambda m: (m.latest_start_time, m.order_index))
        elif self.fusion_strategy == "interleaved":
            # 按最晚开工时间分组
            time_groups: dict[float, list[MatrixFusionInfo]] = {}
            for matrix in self.fused_matrices:
                lst = matrix.latest_start_time
                if lst not in time_groups:
                    time_groups[lst] = []
                time_groups[lst].append(matrix)

            # 对于每个时间组，保持原始顺序作为交错基础
            sorted_matrices = []
            for lst in sorted(time_groups.keys()):
                group = time_groups[lst]
                # 如果组内有多个矩阵，它们会按添加顺序交错执行
                sorted_matrices.extend(group)

            self.fused_matrices = sorted_matrices
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        # 更新 order_index
        for i, matrix in enumerate(self.fused_matrices):
            matrix.order_index = i

    def get_output_nodes(self) -> list[str]:
        """获取所有输出节点名称

        Returns:
            输出节点名称列表，按执行顺序
        """
        return [m.node_name for m in self.fused_matrices]

    def get_matrix_count(self) -> int:
        """获取融合的矩阵数量

        Returns:
            融合的矩阵数量
        """
        return len(self.fused_matrices)


    def __repr__(self) -> str:
        return (
            f"FusionMatrixOp(inputs={self.shared_inputs}, "
            f"matrices={len(self.fused_matrices)}, "
            f"strategy={self.fusion_strategy})"
        )

    def __len__(self) -> int:
        return len(self.fused_matrices)


# 保留 FusionMatrix 作为别名，用于向后兼容
FusionMatrix = FusionMatrixOp


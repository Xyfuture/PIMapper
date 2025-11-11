"""FusionMatrix 操作

用于表示融合后的矩阵运算，记录多个矩阵的融合情况和执行顺序。
使用合并树结构来管理融合策略和输出提取。
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Any, ClassVar, Optional

from pimapper.core.graph.ops.base import GraphTensor, Op, ResultFilter, TensorTransform
from pimapper.core.graph.ops.native import MatMulOp


class FusionStrategy(Enum):
    """融合策略枚举

    SEQUENTIAL: 按照指定顺序将多个矩阵排布，直接规定好谁在前，谁在后
    INTERLEAVED: 将多个矩阵的权重交错分布，沿着列的维度对矩阵进行拆分
    """
    SEQUENTIAL = "sequential"
    INTERLEAVED = "interleaved"


@dataclass
class MergeTreeNode:
    """合并树节点

    合并树用于管理多个矩阵的融合顺序和输出提取。
    - 叶节点：存储原始的 Op 信息
    - 内部节点：存储融合策略，描述子节点如何合并

    Attributes:
        is_leaf: 是否为叶节点
        op_name: 原始 Op 的节点名称（仅叶节点）
        op: 原始 Op 对象（仅叶节点）
        output_shape: 输出张量形状 (batch_dims..., rows, cols)
        weight_shape: 权重矩阵形状 (rows, cols)，用于硬件映射
        latest_start_time: 最晚开工时间
        strategy: 融合策略（仅内部节点）
        children: 子节点列表（仅内部节点）
        col_offset: 在融合矩阵中的列偏移量
        col_size: 在融合矩阵中占据的列数
        block_size: 交错模式下的块大小（仅 INTERLEAVED 策略）
        block_offsets: 交错模式下，每个块在融合矩阵中的偏移量列表 [(offset, size), ...]
    """
    is_leaf: bool
    op_name: Optional[str] = None
    op: Optional[Op] = None
    output_shape: Optional[tuple[int, ...]] = None
    weight_shape: Optional[tuple[int, int]] = None
    latest_start_time: float = 0.0
    strategy: Optional[FusionStrategy] = None
    children: list[MergeTreeNode] = dataclass_field(default_factory=list)
    col_offset: int = 0
    col_size: int = 0
    block_size: int = 1
    block_offsets: list[tuple[int, int]] = dataclass_field(default_factory=list)

    @classmethod
    def create_leaf(
        cls,
        op_name: str,
        op: Op,
        shape: tuple[int, ...],
        latest_start_time: float,
    ) -> MergeTreeNode:
        """创建叶节点

        Args:
            op_name: 原始 Op 的节点名称
            op: 原始 Op 对象
            shape: 输出张量形状 (batch_dims..., rows, cols)
            latest_start_time: 最晚开工时间

        Returns:
            叶节点
        """
        # 提取权重矩阵形状 (rows, cols)
        weight_shape = None

        # 首先尝试从 metadata 中提取权重形状（对于 Linear 层）
        if hasattr(op, 'metadata') and op.metadata:
            custom = op.metadata.get('custom', {})
            metadata_weight_shape = custom.get('weight_shape')

            if metadata_weight_shape is not None:
                # metadata 中的 weight_shape 是 (out_features, in_features)
                # 检查是否需要转置
                transpose_b = op.kwargs.get('transpose_b', False) if hasattr(op, 'kwargs') else False

                if transpose_b:
                    # 转置：(out_features, in_features) -> (in_features, out_features)
                    weight_shape = (metadata_weight_shape[1], metadata_weight_shape[0])
                else:
                    weight_shape = metadata_weight_shape

        # 如果 metadata 中没有，回退到从输出形状提取（不准确）
        if weight_shape is None and shape and len(shape) >= 2:
            weight_shape = (shape[-2], shape[-1])

        # col_size 是权重矩阵的列数
        col_size = weight_shape[1] if weight_shape else (shape[-1] if shape and len(shape) >= 2 else 0)

        return cls(
            is_leaf=True,
            op_name=op_name,
            op=op,
            output_shape=shape,
            weight_shape=weight_shape,
            latest_start_time=latest_start_time,
            col_size=col_size,
        )

    @classmethod
    def create_internal(
        cls,
        children: list[MergeTreeNode],
        strategy: FusionStrategy,
        block_size: int = 1,
    ) -> MergeTreeNode:
        """创建内部节点

        Args:
            children: 子节点列表
            strategy: 融合策略
            block_size: 交错模式下的块大小

        Returns:
            内部节点
        """
        if not children:
            raise ValueError("Internal node must have at least one child")

        # 计算融合后的形状（假设所有子节点的行数相同）
        total_cols = sum(child.col_size for child in children)
        first_output_shape = children[0].output_shape

        # 计算融合后的输出形状
        if first_output_shape and len(first_output_shape) >= 2:
            fused_output_shape = first_output_shape[:-1] + (total_cols,)
        else:
            fused_output_shape = None

        # 计算融合后的权重矩阵形状 (rows, fused_cols)
        # 从子节点的 weight_shape 中提取行数（而不是从 output_shape）
        fused_weight_shape = None
        first_weight_shape = children[0].weight_shape
        if first_weight_shape is not None:
            rows = first_weight_shape[0]
            fused_weight_shape = (rows, total_cols)
        elif first_output_shape and len(first_output_shape) >= 2:
            # 回退方案：从输出形状提取（不准确）
            rows = first_output_shape[-2]
            fused_weight_shape = (rows, total_cols)

        # 使用子节点中最小的 latest_start_time
        lst = min(child.latest_start_time for child in children)

        return cls(
            is_leaf=False,
            output_shape=fused_output_shape,
            weight_shape=fused_weight_shape,
            latest_start_time=lst,
            strategy=strategy,
            children=children,
            col_size=total_cols,
            block_size=block_size,
        )

    def compute_offsets(self, start_offset: int = 0) -> None:
        """计算所有节点的列偏移量

        从根节点向下递归计算每个节点在融合矩阵中的列偏移量。

        Args:
            start_offset: 起始偏移量
        """
        self.col_offset = start_offset

        if not self.is_leaf and self.children:
            if self.strategy == FusionStrategy.SEQUENTIAL:
                # 顺序排布：子节点依次排列
                current_offset = start_offset
                for child in self.children:
                    child.compute_offsets(current_offset)
                    # 对于顺序模式，记录连续的块偏移量
                    child.block_offsets = [(current_offset, child.col_size)]
                    current_offset += child.col_size

            elif self.strategy == FusionStrategy.INTERLEAVED:
                # 交错排布：按块交错分布
                # 计算每个子节点的块数
                block_counts = [
                    (child.col_size + self.block_size - 1) // self.block_size
                    for child in self.children
                ]

                # 为每个子节点分配偏移量列表
                child_block_offsets = [[] for _ in self.children]
                current_offset = start_offset

                # 交错分配块
                max_blocks = max(block_counts)
                for block_idx in range(max_blocks):
                    for child_idx, child in enumerate(self.children):
                        if block_idx < block_counts[child_idx]:
                            # 最后一个块可能不完整
                            remaining_cols = child.col_size - block_idx * self.block_size
                            block_cols = min(self.block_size, remaining_cols)
                            child_block_offsets[child_idx].append((current_offset, block_cols))
                            current_offset += block_cols

                # 递归计算子节点的偏移量（对于交错模式，子节点需要特殊处理）
                for child_idx, child in enumerate(self.children):
                    # 对于交错模式，子节点的 col_offset 是第一个块的偏移量
                    child.block_offsets = child_block_offsets[child_idx]
                    if child.block_offsets:
                        child.col_offset = child.block_offsets[0][0]
                    if not child.is_leaf:
                        child.compute_offsets(child.col_offset)

    def get_result_filter(self, target_op_name: str) -> Optional[ResultFilter]:
        """获取指定 Op 的输出提取过滤器

        从融合矩阵的输出中提取出对应原始 Op 的输出。
        采用从 root 到 leaf 的遍历方式，每经过一个 internal node 就添加一个 TensorTransform。

        Args:
            target_op_name: 目标 Op 的节点名称

        Returns:
            ResultFilter 对象，如果找不到则返回 None
        """
        # 在根节点创建 ResultFilter
        filter = ResultFilter(index=0)

        # 递归查找并构建变换链
        if self._build_filter_chain(target_op_name, filter):
            return filter
        else:
            return None

    def _build_filter_chain(self, target_op_name: str, filter: ResultFilter) -> bool:
        """递归构建过滤器变换链

        从当前节点向下查找目标 Op，每经过一个 internal node 就添加相应的变换。

        Args:
            target_op_name: 目标 Op 的节点名称
            filter: 要构建的 ResultFilter 对象

        Returns:
            True 如果找到目标 Op，False 否则
        """
        if self.is_leaf:
            # 叶节点：检查是否是目标 Op
            return self.op_name == target_op_name
        else:
            # 内部节点：查找哪个子节点包含目标 Op
            for child_idx, child in enumerate(self.children):
                if child._contains_op(target_op_name):
                    # 找到了包含目标 Op 的子节点
                    # 添加从当前节点提取该子节点输出的变换
                    self._add_extraction_transform(filter, child_idx)

                    # 递归到子节点继续构建
                    return child._build_filter_chain(target_op_name, filter)

            return False

    def _contains_op(self, target_op_name: str) -> bool:
        """检查当前节点是否包含目标 Op

        Args:
            target_op_name: 目标 Op 的节点名称

        Returns:
            True 如果包含，False 否则
        """
        if self.is_leaf:
            return self.op_name == target_op_name
        else:
            return any(child._contains_op(target_op_name) for child in self.children)

    def _add_extraction_transform(self, filter: ResultFilter, child_idx: int) -> None:
        """添加从当前节点输出中提取指定子节点结果的变换

        Args:
            filter: 要添加变换的 ResultFilter
            child_idx: 子节点索引
        """
        child = self.children[child_idx]

        if self.strategy == FusionStrategy.SEQUENTIAL:
            # SEQUENTIAL 模式：子节点的输出是连续的
            # 使用简单的 slice [start:end]
            start = child.col_offset
            end = child.col_offset + child.col_size
            filter.add_transform(
                TensorTransform.slice(
                    (Ellipsis, slice(start, end))
                )
            )

        elif self.strategy == FusionStrategy.INTERLEAVED:
            # INTERLEAVED 模式：子节点的输出是交错的
            # 需要使用 strided_slice 提取

            # 对于每个子节点，计算其在每轮中的起始位置
            start_in_round = sum(
                min(self.block_size, self.children[i].col_size)
                for i in range(child_idx)
            )

            # 计算总的 stride（一轮的总大小）
            round_size = sum(
                min(self.block_size, c.col_size)
                for c in self.children
            )

            # 子节点的块大小
            child_block_size = min(self.block_size, child.col_size)

            # 计算起始和结束位置
            start = self.col_offset + start_in_round
            # 结束位置需要覆盖所有块
            num_blocks = (child.col_size + self.block_size - 1) // self.block_size
            end = start + (num_blocks - 1) * round_size + child_block_size

            filter.add_transform(
                TensorTransform.strided_slice(
                    start=start,
                    end=end,
                    stride=round_size,
                    block_size=child_block_size,
                    dim=-1
                )
            )

    def get_all_leaf_ops(self) -> list[tuple[str, Op]]:
        """获取所有叶节点的 Op

        Returns:
            (op_name, op) 元组列表
        """
        if self.is_leaf:
            if self.op_name is not None and self.op is not None:
                return [(self.op_name, self.op)]
            else:
                return []
        else:
            result = []
            for child in self.children:
                result.extend(child.get_all_leaf_ops())
            return result


class FusionMatrixOp(MatMulOp):
    """融合矩阵操作

    将多个输入相同的矩阵运算融合为一个操作，使用合并树管理融合顺序。
    继承自 MatMulOp，可以被 MatrixMappingPass 识别和映射。

    Attributes:
        merge_tree: 合并树根节点
        shared_inputs: 共享的输入节点名称列表
        fused_weight_shape: 融合后的权重矩阵形状 (rows, cols)，用于硬件映射
    """

    op_type: ClassVar[str] = "fusion_matrix"

    def __init__(
        self,
        merge_tree: MergeTreeNode,
        shared_inputs: list[str],
        args: Optional[tuple[Any, ...]] = None,
        kwargs: Optional[dict[str, Any]] = None,
    ):
        """初始化融合矩阵操作

        Args:
            merge_tree: 合并树根节点
            shared_inputs: 共享的输入节点名称列表
            args: 操作参数
            kwargs: 操作关键字参数
        """
        # 计算偏移量
        merge_tree.compute_offsets(0)

        # 提取融合后的权重矩阵形状
        fused_weight_shape = merge_tree.weight_shape

        # 准备 kwargs，包含融合矩阵的特殊信息
        fusion_kwargs = kwargs or {}
        if fused_weight_shape:
            fusion_kwargs['matrix_shape'] = fused_weight_shape

        # 调用父类构造函数
        super().__init__(
            transpose_a=fusion_kwargs.get('transpose_a', False),
            transpose_b=fusion_kwargs.get('transpose_b', False),
            matrix_shape=fused_weight_shape
        )

        # 覆盖 args（使用共享输入）
        self.args = args or tuple(shared_inputs)

        # 存储融合特定的属性
        self.merge_tree = merge_tree
        self.shared_inputs = shared_inputs
        self.fused_weight_shape = fused_weight_shape

        # 设置 results：融合后的输出形状
        if merge_tree.output_shape:
            self.results = [GraphTensor(shape=merge_tree.output_shape)]
        else:
            self.results = []

    def get_result_filter(self, target_op_name: str) -> Optional[ResultFilter]:
        """获取指定 Op 的输出提取过滤器

        Args:
            target_op_name: 目标 Op 的节点名称

        Returns:
            ResultFilter 对象，如果找不到则返回 None
        """
        return self.merge_tree.get_result_filter(target_op_name)

    def get_all_original_ops(self) -> list[tuple[str, Op]]:
        """获取所有原始 Op

        Returns:
            (op_name, op) 元组列表
        """
        return self.merge_tree.get_all_leaf_ops()

    def __repr__(self) -> str:
        leaf_count = len(self.merge_tree.get_all_leaf_ops())
        return (
            f"FusionMatrixOp(inputs={self.shared_inputs}, "
            f"matrices={leaf_count}, "
            f"shape={self.merge_tree.shape})"
        )


# 保留 FusionMatrix 作为别名，用于向后兼容
FusionMatrix = FusionMatrixOp

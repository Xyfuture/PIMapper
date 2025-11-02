"""操作归一化 Pass

将 torch 风格的操作转换为原生操作。提供注册系统，让原生 op 类自动注册转换规则。
"""

from __future__ import annotations

from typing import Any, Type

from pimapper.core.graph.base import NxComputationGraph
from pimapper.core.graph.ops.base import Op
from pimapper.core.graph.ops.torch_compat import TorchFxOp
from pimapper.modelmapper.passes.base import Pass


class NormalizeOpsPass(Pass):
    """操作归一化 Pass

    将计算图中的 torch 风格操作（call_function, call_method, call_module）
    转换为原生操作（matmul, vector_add, silu 等）。

    工作流程：
    1. 遍历计算图中的所有节点
    2. 对于每个 torch 操作节点，查找可以转换的原生 op 类
    3. 调用原生 op 类的 convert_from_torch 方法进行转换
    4. 更新节点属性为原生 op 的属性

    注册系统：
    - 原生 op 类通过 register_op() 方法注册
    - 注册时会调用 can_convert_from_torch() 判断是否可以转换
    """

    # 类级别的 op 注册表
    _registered_ops: list[Type[Op]] = []

    def __init__(self, *, name: str | None = None):
        """初始化操作归一化 Pass

        Args:
            name: Pass 名称（可选）
        """
        super().__init__(
            name=name or "NormalizeOps",
            description="Convert torch operations to native operations",
        )

    @classmethod
    def register_op(cls, op_class: Type[Op]) -> None:
        """注册原生 op 类

        将原生 op 类添加到注册表中，用于自动转换。

        Args:
            op_class: 要注册的 op 类
        """
        if op_class not in cls._registered_ops:
            cls._registered_ops.append(op_class)

    @classmethod
    def unregister_op(cls, op_class: Type[Op]) -> None:
        """取消注册原生 op 类

        Args:
            op_class: 要取消注册的 op 类
        """
        if op_class in cls._registered_ops:
            cls._registered_ops.remove(op_class)

    @classmethod
    def clear_registry(cls) -> None:
        """清空注册表"""
        cls._registered_ops.clear()

    @classmethod
    def get_registered_ops(cls) -> list[Type[Op]]:
        """获取所有已注册的 op 类

        Returns:
            已注册的 op 类列表
        """
        return cls._registered_ops.copy()

    def run(self, graph: NxComputationGraph) -> bool:
        """执行操作归一化

        Args:
            graph: 要处理的计算图

        Returns:
            True 如果有操作被转换，False 如果没有改变
        """
        converted_count = 0
        skipped_count = 0
        failed_count = 0

        # 遍历所有节点
        for node_name in graph.nodes(sort=False):
            op = graph.node_record(node_name)

            # 跳过非 torch 操作节点
            if not isinstance(op, TorchFxOp):
                continue

            # 尝试转换为原生操作
            converted_op = self._try_convert_op(op)

            if converted_op is None:
                skipped_count += 1
                continue

            # 更新节点属性
            try:
                self._update_node_with_op(graph, node_name, converted_op)
                converted_count += 1
            except Exception as e:
                self._metadata.setdefault("errors", []).append(
                    f"Failed to update node '{node_name}': {str(e)}"
                )
                failed_count += 1

        # 更新统计信息
        self._metadata["converted_count"] = converted_count
        self._metadata["skipped_count"] = skipped_count
        self._metadata["failed_count"] = failed_count
        self._metadata["total_checked"] = converted_count + skipped_count + failed_count

        return converted_count > 0

    def _try_convert_op(self, op: TorchFxOp) -> Op | None:
        """尝试将 Op 转换为原生操作"""
        for op_class in self._registered_ops:
            try:
                native_op = op_class.convert_from_torch(op)
                # 保留原 op 的 results
                native_op.results = op.results
                return native_op
            except Exception:
                continue
        return None

    def _update_node_with_op(
        self, graph: NxComputationGraph, node_name: str, op: Op
    ) -> None:
        """使用原生 op 替换节点的 Op 对象

        Args:
            graph: 计算图
            node_name: 节点名称
            op: 原生 op 实例
        """
        # Simply replace the op object in the node data
        graph.graph.nodes[node_name]["op"] = op


def auto_register_native_ops() -> None:
    """自动注册所有原生操作

    导入并注册 native.py 中定义的所有原生 op 类。
    """
    from pimapper.core.graph.ops.native import NATIVE_OPS

    for op_class in NATIVE_OPS:
        NormalizeOpsPass.register_op(op_class)


# 模块加载时自动注册
auto_register_native_ops()

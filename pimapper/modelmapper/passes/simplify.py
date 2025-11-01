"""计算图简化 Pass

移除计算图中的非核心操作，保留重要的计算节点。
这个 Pass 将 converter.py 中的 simplify_computation_graph 函数重构为标准的 Pass 格式。
"""

from __future__ import annotations

from typing import Any

from pimapper.core.graph.base import NxComputationGraph
from pimapper.modelmapper.passes.base import Pass


class SimplifyGraphPass(Pass):
    """计算图简化优化 Pass

    移除计算图中的非核心操作，保留重要的计算节点。主要功能：

    保留的操作类型：
    1. 原生操作 (native ops):
       - matmul: 矩阵乘法
       - vector_add: 向量加法
       - vector_dot: 向量点积
       - silu: SiLU 激活函数
       - softmax: Softmax 归一化
       - rmsnorm: RMS 归一化
    2. 神经网络模块 (call_module): 线性层、注意力层、归一化层等核心计算模块
    3. 核心张量操作 (call_function):
       - 激活函数: silu, sigmoid, tanh, relu, gelu
       - 基础线性代数: add, sub, mul, div, matmul, bmm
       - 向量操作: sum, mean, max, min
       - RoPE相关: sin, cos (旋转位置编码)
    4. 方法调用 (call_method): 张量的核心运算方法
    5. 输入占位符 (placeholder): 模型输入节点

    过滤掉的操作：
    - 视图操作: view, reshape, flatten, squeeze, unsqueeze
    - 扩展操作: expand, repeat, tile
    - 形状变换操作: transpose, permute, contiguous
    - 属性访问操作: getattr, getitem, setitem

    这些操作不改变实际数据，只是改变张量的视图或形状。
    """

    def __init__(self, *, name: str | None = None):
        """初始化简化图 Pass

        Args:
            name: Pass 名称（可选）
        """
        super().__init__(
            name=name or "SimplifyGraph",
            description="Remove non-essential operations from computation graph"
        )

        # 定义需要保留的原生操作类型
        self.NATIVE_OPERATIONS = {
            'matmul',
            'vector_add',
            'vector_mul',
            'vector_dot',
            'silu',
            'softmax',
            'rmsnorm',
        }

        # 定义需要保留的核心操作类型（torch 兼容）
        self.ESSENTIAL_OPERATIONS = {
            'call_module',     # 神经网络模块
            'call_function',   # 核心张量操作
            'call_method',     # 方法调用
            'output',          # 输出节点
            'placeholder',     # 输入占位符
        }

        # 定义需要保留的具体函数
        self.ESSENTIAL_FUNCTIONS = {
            # 激活函数
            'silu', 'sigmoid', 'tanh', 'relu', 'gelu',
            # 基础线性代数运算
            'add', 'sub', 'mul', 'div', 'matmul', 'bmm',
            # 向量聚合操作
            'sum', 'mean', 'max', 'min',
            # RoPE (旋转位置编码) 相关三角函数
            'sin', 'cos',
        }

        # 定义需要保留的特定方法
        self.ESSENTIAL_METHODS = {
            # 张量运算方法
            'add', 'sub', 'mul', 'div', 'matmul',
            # 聚合方法
            'sum', 'mean', 'max', 'min',
        }

        # 定义需要过滤掉的操作
        self.FILTERED_OPERATIONS = {
            'view', 'reshape', 'flatten', 'squeeze', 'unsqueeze',  # 视图操作
            'expand', 'repeat', 'tile',  # 扩展和重复操作
            'transpose', 'permute', 'contiguous',  # 形状变换操作
            'getattr', 'getitem', 'setitem',  # 属性访问和索引操作
        }

    def run(self, graph: NxComputationGraph) -> bool:
        """执行计算图简化

        Args:
            graph: 要简化的计算图

        Returns:
            True 如果图被修改，False 如果图未改变
        """
        # 第一遍：标记需要删除的节点
        nodes_to_remove = []
        for node_name in graph.nodes(sort=False):
            if not self._should_keep_node(graph, node_name):
                nodes_to_remove.append(node_name)

        if not nodes_to_remove:
            self._metadata["nodes_removed"] = 0
            self._metadata["nodes_examined"] = len(list(graph.nodes()))
            return False

        # 第二遍：安全删除节点
        removed_count = 0
        for node_name in nodes_to_remove:
            if node_name not in graph.graph:  # 节点可能已经被删除
                continue

            if self._remove_node_safely(graph, node_name):
                removed_count += 1

        # 更新统计信息
        self._metadata["nodes_removed"] = removed_count
        self._metadata["nodes_examined"] = len(list(graph.nodes())) + removed_count
        self._metadata["original_node_count"] = self._metadata["nodes_examined"]
        self._metadata["final_node_count"] = len(list(graph.nodes()))

        return removed_count > 0

    def _should_keep_node(self, graph: NxComputationGraph, node_name: str) -> bool:
        """判断节点是否应该保留

        Args:
            graph: 计算图
            node_name: 节点名称

        Returns:
            True 如果节点应该保留，False 如果应该删除
        """
        op = graph.node_record(node_name)
        op_type = op.op_type
        target = getattr(op, 'target', None)

        # 检查是否为原生操作（优先检查）
        if op_type in self.NATIVE_OPERATIONS:
            return True

        # 检查是否为核心操作类型
        if op_type not in self.ESSENTIAL_OPERATIONS:
            return False

        if op_type == 'call_module':
            # 始终保留模块，但过滤掉视图类模块
            module_name = getattr(op, 'module_name', None) or target
            if isinstance(module_name, str) and any(
                filtered in module_name.lower() for filtered in self.FILTERED_OPERATIONS
            ):
                return False
            return True

        elif op_type == 'call_function':
            target_str = str(target).lower() if target else ''

            # 检查是否为保留的特定函数
            if any(essential in target_str for essential in self.ESSENTIAL_FUNCTIONS):
                return True

            # 检查是否包含过滤操作
            if any(filtered in target_str for filtered in self.FILTERED_OPERATIONS):
                return False

            # 其他函数默认过滤
            return False

        elif op_type == 'call_method':
            method_name = getattr(op, 'method_name', None) or target
            target_str = str(method_name).lower() if method_name else ''

            # 检查是否为保留的特定方法
            if any(essential in target_str for essential in self.ESSENTIAL_METHODS):
                return True

            # 检查是否包含过滤操作
            if any(filtered in target_str for filtered in self.FILTERED_OPERATIONS):
                return False

            # 其他方法默认过滤
            return False

        elif op_type == 'placeholder':
            return True

        elif op_type == 'output':
            return True

        return False

    def _remove_node_safely(self, graph: NxComputationGraph, node_name: str) -> bool:
        """安全删除节点

        根据节点的连接情况选择合适的删除策略：
        - 无前驱节点：直接删除
        - 单个前驱节点：使用 replace_uses 重定向后继节点，然后删除
        - 多个前驱节点：暂不支持，返回 False

        Args:
            graph: 计算图
            node_name: 要删除的节点名称

        Returns:
            True 如果成功删除，False 否则
        """
        op = graph.node_record(node_name)
        op_type = op.op_type
        target = getattr(op, 'target', None)

        # 获取节点的前驱和后继
        predecessors = list(graph.predecessors(node_name))
        successors = list(graph.successors(node_name))

        # 对于属性访问操作，直接删除
        if op_type == 'call_function' and any(
            filtered in str(target).lower() for filtered in {'getattr', 'getitem', 'setitem'}
        ):
            graph.remove_node(node_name, safe=False)
            return True

        # 无前驱节点：直接删除
        if len(predecessors) == 0:
            graph.remove_node(node_name, safe=False)
            return True

        # 单个前驱节点：重定向后继节点
        elif len(predecessors) == 1:
            graph.replace_uses(node_name, predecessors[0])
            graph.remove_node(node_name, safe=False)
            return True

        # 多个前驱节点：暂不支持
        else:
            # 这种情况需要更复杂的逻辑，暂不处理
            return False
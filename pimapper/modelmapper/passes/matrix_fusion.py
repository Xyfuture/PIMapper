"""矩阵融合 Pass

将共享输入的矩阵运算融合，使用最晚开工时间算法优化执行顺序。
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from pimapper.core.graph.base import NxComputationGraph
from pimapper.core.graph.ops.fusionmatrix import FusionMatrix
from pimapper.modelmapper.passes.base import Pass


class MatrixFusionPass(Pass):
    """矩阵融合优化 Pass

    识别共享相同输入的矩阵运算，并将它们融合为一个 FusionMatrix 节点。
    使用最晚开工时间（Latest Start Time）算法来确定融合后的执行顺序。

    算法流程：
    1. 识别计算图中的矩阵运算节点（根据形状信息）
    2. 找到共享输入的矩阵运算组
    3. 计算每个节点的最晚开工时间
    4. 为每组创建融合节点，按最晚开工时间排序
    5. 改写计算图，替换原始节点

    时间计算假设：
    - 矩阵运算（2D+张量运算）：时间 = 1
    - 向量运算（1D 或标量运算）：时间 = 0
    """

    def __init__(
        self,
        fusion_strategy: str = "sequential",
        min_fusion_size: int = 2,
        *,
        name: str | None = None,
    ):
        """初始化矩阵融合 Pass

        Args:
            fusion_strategy: 融合策略 ('sequential' 或 'interleaved')
            min_fusion_size: 最小融合大小，少于此数量的不融合
            name: Pass 名称（可选）
        """
        super().__init__(name=name or "MatrixFusion", description="Fuse matrices with shared inputs")
        self.fusion_strategy = fusion_strategy
        self.min_fusion_size = min_fusion_size

    def run(self, graph: NxComputationGraph) -> bool:
        """执行矩阵融合

        Args:
            graph: 要优化的计算图

        Returns:
            True 如果图被修改，False 如果图未改变
        """
        # 1. 识别矩阵运算节点
        matrix_nodes = self._identify_matrix_ops(graph)

        if not matrix_nodes:
            self._metadata["matrix_nodes_found"] = 0
            self._metadata["fusion_groups"] = 0
            return False

        # 2. 按共享输入分组
        fusion_groups = self._group_by_shared_inputs(graph, matrix_nodes)

        # 过滤掉太小的组
        fusion_groups = {
            inputs: nodes for inputs, nodes in fusion_groups.items() if len(nodes) >= self.min_fusion_size
        }

        if not fusion_groups:
            self._metadata["matrix_nodes_found"] = len(matrix_nodes)
            self._metadata["fusion_groups"] = 0
            return False

        # 3. 计算最晚开工时间
        latest_start_times = self._compute_latest_start_times(graph)

        # 4. 对每个融合组创建融合节点
        modified = False
        for shared_inputs, nodes in fusion_groups.items():
            success = self._fuse_matrix_group(graph, nodes, shared_inputs, latest_start_times)
            if success:
                modified = True

        # 更新统计信息
        self._metadata["matrix_nodes_found"] = len(matrix_nodes)
        self._metadata["fusion_groups"] = len(fusion_groups)
        self._metadata["total_fused_nodes"] = sum(len(nodes) for nodes in fusion_groups.values())

        return modified

    def _identify_matrix_ops(self, graph: NxComputationGraph) -> list[str]:
        """识别矩阵运算节点

        根据以下条件判断是否为矩阵运算：
        1. 是 call_function 或 call_method 或 call_module
        2. 有形状信息且输出是 2D 及以上的张量
        3. 不是纯粹的 reshape/view 操作

        Args:
            graph: 计算图

        Returns:
            矩阵运算节点名称列表
        """
        matrix_nodes = []

        for node_name in graph.nodes(sort=True):
            record = graph.node_record(node_name)

            # 必须是函数/方法/模块调用 或者原生矩阵操作
            op_type = getattr(record, 'op', None) or getattr(record, 'op_type', None)
            if op_type not in ("call_function", "call_method", "call_module", "matmul"):
                continue

            # 必须有形状信息
            shape = getattr(record.metadata, 'shape', None) if hasattr(record, 'metadata') else None
            if not shape:
                continue

            # 形状必须是 2D 及以上
            if len(shape) < 2:
                continue

            # 排除 reshape/view 类操作（仅对有target的节点）
            if hasattr(record, 'target'):
                target_str = str(record.target).lower()
                if any(
                    kw in target_str
                    for kw in ["reshape", "view", "flatten", "squeeze", "unsqueeze", "transpose", "permute"]
                ):
                    continue

            matrix_nodes.append(node_name)

        return matrix_nodes

    def _group_by_shared_inputs(
        self, graph: NxComputationGraph, matrix_nodes: list[str]
    ) -> dict[tuple[str, ...], list[str]]:
        """将矩阵运算按共享输入分组

        Args:
            graph: 计算图
            matrix_nodes: 矩阵运算节点列表

        Returns:
            字典，key 是输入节点元组（排序后），value 是共享该输入的节点列表
        """
        groups: dict[tuple[str, ...], list[str]] = {}

        for node_name in matrix_nodes:
            # 获取该节点的所有输入
            predecessors = graph.predecessors(node_name)

            # 过滤出真正的数据输入（排除 get_attr 等）
            data_inputs = []
            for pred in predecessors:
                pred_record = graph.node_record(pred)
                pred_op_type = getattr(pred_record, 'op', None) or getattr(pred_record, 'op_type', None)
                if pred_op_type in ("placeholder", "call_function", "call_method", "call_module", "rmsnorm", "silu", "softmax", "vector_mul"):
                    data_inputs.append(pred)

            if not data_inputs:
                continue

            # 按字母顺序排序作为 key
            inputs_key = tuple(sorted(data_inputs))

            if inputs_key not in groups:
                groups[inputs_key] = []
            groups[inputs_key].append(node_name)

        return groups

    def _compute_latest_start_times(self, graph: NxComputationGraph) -> dict[str, float]:
        """计算每个节点的最晚开工时间

        使用关键路径方法（CPM）的反向遍历：
        1. 首先计算最早开始时间（正向遍历）
        2. 然后计算最晚开工时间（反向遍历）

        时间假设：
        - 矩阵运算：时间 = 1
        - 向量运算：时间 = 0

        Args:
            graph: 计算图

        Returns:
            字典，key 是节点名称，value 是最晚开工时间
        """
        # 第一步：计算最早开始时间（EST）和最早完成时间（EFT）
        earliest_start: dict[str, float] = {}
        earliest_finish: dict[str, float] = {}

        for node_name in nx.topological_sort(graph.graph):
            record = graph.node_record(node_name)
            duration = self._get_node_duration(record)

            # 最早开始时间 = max(所有前驱的最早完成时间)
            predecessors = graph.predecessors(node_name)
            if predecessors:
                est = max(earliest_finish.get(pred, 0.0) for pred in predecessors)
            else:
                est = 0.0

            earliest_start[node_name] = est
            earliest_finish[node_name] = est + duration

        # 第二步：计算最晚完成时间（LFT）和最晚开工时间（LST）
        latest_finish: dict[str, float] = {}
        latest_start: dict[str, float] = {}

        # 获取所有输出节点
        output_nodes = []
        for n in graph.nodes():
            record = graph.node_record(n)
            op_type = getattr(record, 'op', None) or getattr(record, 'op_type', None)
            if op_type == "output":
                output_nodes.append(n)

        # 对于输出节点，最晚完成时间 = 最早完成时间（必须按时完成）
        for node_name in output_nodes:
            latest_finish[node_name] = earliest_finish[node_name]

        # 反向拓扑排序
        for node_name in reversed(list(nx.topological_sort(graph.graph))):
            record = graph.node_record(node_name)
            duration = self._get_node_duration(record)

            # 如果已经设置了 latest_finish（输出节点），跳过
            if node_name not in latest_finish:
                # 最晚完成时间 = min(所有后继的最晚开工时间)
                successors = graph.successors(node_name)
                if successors:
                    lft = min(latest_start.get(succ, float("inf")) for succ in successors)
                else:
                    # 没有后继的非输出节点，使用最早完成时间
                    lft = earliest_finish[node_name]

                latest_finish[node_name] = lft

            # 最晚开工时间 = 最晚完成时间 - 持续时间
            latest_start[node_name] = latest_finish[node_name] - duration

        return latest_start

    def _get_node_duration(self, record: Any) -> float:
        """获取节点的执行时间

        Args:
            record: 节点记录

        Returns:
            执行时间（矩阵运算=1，其他=0）
        """
        # 如果有形状信息且是 2D 及以上，认为是矩阵运算
        shape = getattr(record.metadata, 'shape', None) if hasattr(record, 'metadata') else None
        if shape and len(shape) >= 2:
            return 1.0
        return 0.0

    def _fuse_matrix_group(
        self,
        graph: NxComputationGraph,
        nodes: list[str],
        shared_inputs: tuple[str, ...],
        latest_start_times: dict[str, float],
    ) -> bool:
        """将一组矩阵融合为一个 FusionMatrix 节点

        Args:
            graph: 计算图
            nodes: 要融合的节点列表
            shared_inputs: 共享的输入节点
            latest_start_times: 最晚开工时间字典

        Returns:
            True 如果成功融合，False 否则
        """
        # 创建 FusionMatrix
        fusion = FusionMatrix(list(shared_inputs), fusion_strategy=self.fusion_strategy)

        # 添加所有矩阵到融合操作
        for node_name in nodes:
            record = graph.node_record(node_name)
            lst = latest_start_times.get(node_name, 0.0)
            op_type = getattr(record, 'op', None) or getattr(record, 'op_type', None)

            shape = getattr(record.metadata, 'shape', None) if hasattr(record, 'metadata') else None
            dtype = getattr(record.metadata, 'dtype', None) if hasattr(record, 'metadata') else None
            target = getattr(record, 'target', op_type)
            args = getattr(record, 'args', ())
            kwargs = getattr(record, 'kwargs', {})
            fusion.add_matrix(
                node_name=node_name,
                target=target,
                args=args,
                kwargs=kwargs,
                latest_start_time=lst,
                shape=shape,
                dtype=dtype,
            )

        # 根据最晚开工时间排序
        fusion.sort_matrices()

        # 创建融合节点
        fusion_node_name = self._create_fusion_node(graph, fusion, shared_inputs)

        # 改写计算图，重定向所有使用原始节点的地方到融合节点的相应输出
        self._rewrite_graph(graph, nodes, fusion_node_name, fusion)

        return True

    def _create_fusion_node(
        self, graph: NxComputationGraph, fusion: FusionMatrix, shared_inputs: tuple[str, ...]
    ) -> str:
        """创建融合节点

        Args:
            graph: 计算图
            fusion: 融合矩阵对象
            shared_inputs: 共享输入

        Returns:
            新创建的融合节点名称
        """
        # 创建融合节点
        from pimapper.core.graph.ops.base import OpMetadata
        fusion_node_name = f"fused_matrix_{len(fusion)}"
        graph.create_node(
            name=fusion_node_name,
            op=fusion,
        )

        return fusion_node_name

    def _rewrite_graph(
        self, graph: NxComputationGraph, original_nodes: list[str], fusion_node: str, fusion: FusionMatrix
    ) -> None:
        """改写计算图，用融合节点替换原始节点

        策略：
        1. 直接用融合节点替换所有原始节点
        2. 将所有原始节点的输入连接到融合节点
        3. 将所有原始节点的输出连接到融合节点
        4. 删除原始节点

        Args:
            graph: 计算图
            original_nodes: 原始节点列表
            fusion_node: 融合节点名称
            fusion: 融合矩阵对象
        """
        # 收集所有原始节点的输入和输出连接关系
        all_inputs = set()
        all_successors = set()

        for original_name in original_nodes:
            # 收集原始节点的所有输入
            predecessors = list(graph.predecessors(original_name))
            all_inputs.update(predecessors)

            # 收集原始节点的所有输出（后继节点）
            successors = list(graph.successors(original_name))
            all_successors.update(successors)

        # 为融合节点创建所有必要的输入连接
        for input_node in all_inputs:
            graph.graph.add_edge(input_node, fusion_node)

        # 将原始节点的所有输出连接重定向到融合节点
        for successor in all_successors:
            graph.graph.add_edge(fusion_node, successor)

        # 在后继节点的参数中，将所有原始节点引用替换为融合节点
        for original_name in original_nodes:
            # 将所有使用 original_name 的地方替换为 fusion_node
            graph.replace_uses(original_name, fusion_node)

        # 删除原始节点
        for original_name in original_nodes:
            graph.remove_node(original_name, safe=False)

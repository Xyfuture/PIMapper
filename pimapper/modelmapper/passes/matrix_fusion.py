"""矩阵融合 Pass

将共享输入的矩阵运算融合，使用最晚开工时间算法优化执行顺序。
使用合并树结构管理融合策略和输出提取。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import networkx as nx

from pimapper.core.graph.base import NxComputationGraph
from pimapper.core.graph.ops.base import Op, ResultFilter
from pimapper.core.graph.ops.fusionmatrix import (
    FusionMatrixOp,
    FusionStrategy,
    MergeTreeNode,
)
from pimapper.modelmapper.passes.base import Pass


class MatrixFusionPass(Pass):
    """矩阵融合优化 Pass

    识别共享相同输入的矩阵运算，并将它们融合为一个 FusionMatrixOp 节点。
    使用最晚开工时间（Latest Start Time）算法来确定融合后的执行顺序。

    算法流程：
    1. 识别计算图中的矩阵运算节点（根据 op_type 和形状信息）
    2. 找到共享输入的矩阵运算组
    3. 计算每个节点的最晚开工时间（LST）
    4. 为每组构建合并树，根据 LST 决定融合策略
    5. 创建 FusionMatrixOp 节点并改写计算图

    时间计算假设：
    - 矩阵运算（matmul 等）：时间 = 1
    - 其他运算：时间 = 0

    融合策略选择：
    - 如果多个矩阵的 LST 相同：使用 INTERLEAVED 策略（交错排布）
    - 如果多个矩阵的 LST 不同：使用 SEQUENTIAL 策略（按 LST 升序排布）
    """

    def __init__(
        self,
        min_fusion_size: int = 2,
        block_size: int = 64,
        *,
        name: str | None = None,
    ):
        """初始化矩阵融合 Pass

        Args:
            min_fusion_size: 最小融合大小，少于此数量的不融合
            block_size: 交错模式下的块大小（列数）
            name: Pass 名称（可选）
        """
        super().__init__(name=name or "MatrixFusion", description="Fuse matrices with shared inputs")
        self.min_fusion_size = min_fusion_size
        self.block_size = block_size

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
        1. op_type 是 "matmul" 或其他矩阵运算类型
        2. 有 results 信息且输出是 2D 及以上的张量

        Args:
            graph: 计算图

        Returns:
            矩阵运算节点名称列表
        """
        matrix_nodes = []

        for node_name in graph.nodes(sort=True):
            op = graph.node_record(node_name)

            # 检查 op_type
            op_type = getattr(op, 'op_type', None)
            if op_type not in ("matmul",):
                continue

            # 检查 results
            if not hasattr(op, 'results') or not op.results:
                continue

            # 检查第一个输出的形状
            first_result = op.results[0]
            if not hasattr(first_result, 'shape') or not first_result.shape:
                continue

            # 形状必须是 2D 及以上
            if len(first_result.shape) < 2:
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
            # 获取该节点的所有输入（前驱节点）
            predecessors = list(graph.predecessors(node_name))

            if not predecessors:
                continue

            # 过滤出数据输入（排除 get_attr 等）
            data_inputs = []
            for pred in predecessors:
                pred_op = graph.node_record(pred)
                pred_op_type = getattr(pred_op, 'op_type', None)
                # 排除 get_attr 和 output 节点
                if pred_op_type not in ("get_attr", "output"):
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
        1. 首先计算最早开始时间（EST）和最早完成时间（EFT）
        2. 然后计算最晚完成时间（LFT）和最晚开工时间（LST）

        时间假设：
        - 矩阵运算（matmul）：时间 = 1
        - 其他运算：时间 = 0

        Args:
            graph: 计算图

        Returns:
            字典，key 是节点名称，value 是最晚开工时间
        """
        # 第一步：计算最早开始时间（EST）和最早完成时间（EFT）
        earliest_start: dict[str, float] = {}
        earliest_finish: dict[str, float] = {}

        for node_name in nx.topological_sort(graph.graph):
            op = graph.node_record(node_name)
            duration = self._get_node_duration(op)

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
            op = graph.node_record(n)
            op_type = getattr(op, 'op_type', None)
            if op_type == "output":
                output_nodes.append(n)

        # 如果没有 output 节点，使用所有没有后继的节点
        if not output_nodes:
            for n in graph.nodes():
                if not graph.successors(n):
                    output_nodes.append(n)

        # 对于输出节点，最晚完成时间 = 最早完成时间（必须按时完成）
        for node_name in output_nodes:
            latest_finish[node_name] = earliest_finish[node_name]

        # 反向拓扑排序
        for node_name in reversed(list(nx.topological_sort(graph.graph))):
            op = graph.node_record(node_name)
            duration = self._get_node_duration(op)

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

    def _get_node_duration(self, op: Op) -> float:
        """获取节点的执行时间

        Args:
            op: Op 对象

        Returns:
            执行时间（矩阵运算=1，其他=0）
        """
        op_type = getattr(op, 'op_type', None)
        if op_type == "matmul":
            return 1.0
        return 0.0

    def _fuse_matrix_group(
        self,
        graph: NxComputationGraph,
        nodes: list[str],
        shared_inputs: tuple[str, ...],
        latest_start_times: dict[str, float],
    ) -> bool:
        """将一组矩阵融合为一个 FusionMatrixOp 节点

        Args:
            graph: 计算图
            nodes: 要融合的节点列表
            shared_inputs: 共享的输入节点
            latest_start_times: 最晚开工时间字典

        Returns:
            True 如果成功融合，False 否则
        """
        # 构建合并树
        merge_tree = self._build_merge_tree(graph, nodes, latest_start_times)

        if merge_tree is None:
            return False

        # 创建 FusionMatrixOp
        fusion_op = FusionMatrixOp(
            merge_tree=merge_tree,
            shared_inputs=list(shared_inputs),
        )

        # 创建融合节点
        fusion_node_name = f"fused_matrix_{id(fusion_op)}"
        graph.create_node(name=fusion_node_name, op=fusion_op)

        # 改写计算图
        self._rewrite_graph(graph, nodes, fusion_node_name, fusion_op)

        return True

    def _build_merge_tree(
        self,
        graph: NxComputationGraph,
        nodes: list[str],
        latest_start_times: dict[str, float],
    ) -> MergeTreeNode | None:
        """构建合并树

        根据最晚开工时间（LST）决定融合策略：
        - 如果所有节点的 LST 相同：使用 INTERLEAVED 策略
        - 如果节点的 LST 不同：按 LST 分组，递归构建树

        Args:
            graph: 计算图
            nodes: 要融合的节点列表
            latest_start_times: 最晚开工时间字典

        Returns:
            合并树根节点，如果无法构建则返回 None
        """
        if not nodes:
            return None

        # 如果只有一个节点，创建叶节点
        if len(nodes) == 1:
            node_name = nodes[0]
            op = graph.node_record(node_name)
            lst = latest_start_times.get(node_name, 0.0)

            # 获取形状
            shape = None
            if hasattr(op, 'results') and op.results:
                first_result = op.results[0]
                shape = getattr(first_result, 'shape', None)

            # 确保 shape 不为 None
            if shape is None:
                return None

            return MergeTreeNode.create_leaf(
                op_name=node_name,
                op=op,
                shape=shape,
                latest_start_time=lst,
            )

        # 按 LST 分组
        lst_groups: dict[float, list[str]] = {}
        for node_name in nodes:
            lst = latest_start_times.get(node_name, 0.0)
            if lst not in lst_groups:
                lst_groups[lst] = []
            lst_groups[lst].append(node_name)

        # 如果所有节点的 LST 相同，使用 INTERLEAVED 策略
        if len(lst_groups) == 1:
            children = []
            for node_name in nodes:
                child = self._build_merge_tree(graph, [node_name], latest_start_times)
                if child:
                    children.append(child)

            if not children:
                return None

            return MergeTreeNode.create_internal(
                children=children,
                strategy=FusionStrategy.INTERLEAVED,
                block_size=self.block_size,
            )

        # 如果节点的 LST 不同，使用 SEQUENTIAL 策略
        # 按 LST 升序排序（最早开始的在前面）
        sorted_lsts = sorted(lst_groups.keys())

        children = []
        for lst in sorted_lsts:
            group_nodes = lst_groups[lst]
            child = self._build_merge_tree(graph, group_nodes, latest_start_times)
            if child:
                children.append(child)

        if not children:
            return None

        return MergeTreeNode.create_internal(
            children=children,
            strategy=FusionStrategy.SEQUENTIAL,
            block_size=self.block_size,
        )

    def _rewrite_graph(
        self,
        graph: NxComputationGraph,
        original_nodes: list[str],
        fusion_node: str,
        fusion_op: FusionMatrixOp,
    ) -> None:
        """改写计算图，用融合节点替换原始节点

        策略：
        1. 对于每个原始节点的后继节点，更新其输入引用
        2. 使用 ResultFilter 从融合节点的输出中提取对应的部分
        3. 删除原始节点

        Args:
            graph: 计算图
            original_nodes: 原始节点列表
            fusion_node: 融合节点名称
            fusion_op: 融合操作对象
        """
        # 收集所有原始节点的后继节点及其使用关系
        node_successors: dict[str, list[str]] = {}
        for original_name in original_nodes:
            successors = list(graph.successors(original_name))
            node_successors[original_name] = successors

        # 对于每个原始节点，更新其后继节点的输入
        for original_name in original_nodes:
            successors = node_successors[original_name]

            # 获取该原始节点对应的 ResultFilter
            result_filter = fusion_op.get_result_filter(original_name)

            if result_filter is None:
                # 如果找不到对应的 ResultFilter，跳过
                continue

            # 更新所有后继节点
            for succ_name in successors:
                succ_op = graph.node_record(succ_name)

                # 更新 input_ops
                if hasattr(succ_op, 'input_ops'):
                    # 查找并替换 input_ops 中的引用
                    new_input_ops = OrderedDict()
                    for input_op, input_filter in succ_op.input_ops.items():
                        # 检查 input_op 是否是原始节点
                        input_op_name = None
                        for orig_name in original_nodes:
                            orig_op = graph.node_record(orig_name)
                            if input_op is orig_op:
                                input_op_name = orig_name
                                break

                        if input_op_name == original_name:
                            # 替换为融合节点，并组合 ResultFilter
                            combined_filter = self._combine_filters(result_filter, input_filter)
                            new_input_ops[fusion_op] = combined_filter
                        else:
                            new_input_ops[input_op] = input_filter

                    succ_op.input_ops = new_input_ops

                # 更新 args 和 kwargs 中的引用
                succ_op.args = self._replace_in_args(succ_op.args, original_name, fusion_node)
                succ_op.kwargs = self._replace_in_args(succ_op.kwargs, original_name, fusion_node)

                # 更新边连接
                if graph.graph.has_edge(original_name, succ_name):
                    graph.graph.remove_edge(original_name, succ_name)
                if not graph.graph.has_edge(fusion_node, succ_name):
                    graph.graph.add_edge(fusion_node, succ_name)

        # 删除原始节点
        for original_name in original_nodes:
            graph.remove_node(original_name, safe=False)

    def _combine_filters(self, filter1: ResultFilter, filter2: ResultFilter) -> ResultFilter:
        """组合两个 ResultFilter

        将 filter2 的变换应用到 filter1 的结果上。

        Args:
            filter1: 第一个过滤器（从融合节点提取）
            filter2: 第二个过滤器（原始的输入过滤器）

        Returns:
            组合后的过滤器
        """
        # 创建新的 ResultFilter，使用 filter1 的 index
        combined = ResultFilter(index=filter1.index)

        # 添加 filter1 的所有变换
        for transform in filter1.transforms:
            combined.add_transform(transform)

        # 添加 filter2 的所有变换
        for transform in filter2.transforms:
            combined.add_transform(transform)

        return combined

    def _replace_in_args(self, arg: Any, old: str, new: str) -> Any:
        """递归替换参数中的节点引用

        Args:
            arg: 要处理的参数
            old: 要替换的旧节点名称
            new: 新的节点名称

        Returns:
            替换后的参数结构
        """
        if isinstance(arg, tuple):
            return tuple(self._replace_in_args(a, old, new) for a in arg)
        if isinstance(arg, list):
            return [self._replace_in_args(a, old, new) for a in arg]
        if isinstance(arg, dict):
            return {k: self._replace_in_args(v, old, new) for k, v in arg.items()}
        if arg == old:
            return new
        return arg

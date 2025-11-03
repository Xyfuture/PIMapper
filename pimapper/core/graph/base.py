from __future__ import annotations

from typing import Any, Iterable, Optional

import networkx as nx

from pimapper.core.graph.ops.base import Op


class NxComputationGraph:
    """基于 NetworkX 的计算图实现

    使用 NetworkX.DiGraph 作为底层存储，节点数据存储为 Op 对象。
    主要设计特点：
    1. 节点使用字符串标识符，数据存储为 Op 实例
    2. 边关系自动从 Op 的 args/kwargs 中推断和创建
    3. 支持图算法分析和序列化
    4. 与 Op 基类体系集成

    适用场景：
    - 需要图算法分析的场景
    - 要求高可序列化的应用
    - 与 NetworkX 生态集成的项目
    """

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self._order: int = 0
        self._name_counts: dict[str, int] = {}

    # --------------------------------------------------------------------- utils
    def _next_order(self) -> int:
        """生成下一个节点序号

        用于维护节点的创建顺序，确保拓扑排序的一致性。
        每次调用都会递增内部计数器。

        Returns:
            下一个可用的序号
        """
        self._order += 1
        return self._order

    def _unique_name(self, base: str) -> str:
        """生成唯一的节点名称

        处理名称冲突，确保每个节点都有唯一标识符。
        策略：
        1. 将点号替换为下划线（符合命名规范）
        2. 如果基础名称未被使用，直接返回
        3. 否则添加数字后缀，如 "add_1", "add_2"

        Args:
            base: 期望的基础名称

        Returns:
            唯一的节点名称
        """
        base = base.replace(".", "_")
        count = self._name_counts.get(base, 0)
        if count == 0:
            self._name_counts[base] = 1
            candidate = base
        else:
            candidate = f"{base}_{count}"
            self._name_counts[base] = count + 1
        while candidate in self.graph:
            candidate = f"{base}_{self._name_counts[base]}"
            self._name_counts[base] += 1
        return candidate

    def _replace_in_args(self, arg: Any, old: str, new: str) -> Any:
        """递归替换参数中的节点引用

        在参数结构中查找并替换节点名称引用。用于节点替换操作时
        更新依赖关系。

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

    # ------------------------------------------------------------------ creation
    def create_node(
        self,
        name: str,
        op: Op,
    ) -> str:
        """创建新节点并自动建立数据流边

        核心节点创建方法，接受 Op 对象作为节点数据。自动处理：
        1. 从 Op 中提取输入引用
        2. 建立从输入节点到当前节点的边
        3. 存储 Op 对象作为节点数据

        Args:
            name: 节点名称（必须唯一）
            op: Op 对象，包含操作的所有信息

        Returns:
            创建的节点名称

        Raises:
            ValueError: 当指定名称已存在时
        """
        if name in self.graph:
            raise ValueError(f"Node '{name}' already exists")

        # Store Op object as node data along with order
        order = self._next_order()
        self.graph.add_node(name, op=op, order=order)

        # Create edges from input references in args/kwargs
        input_refs = []
        def extract_refs(obj):
            if isinstance(obj, str):
                input_refs.append(obj)
            elif isinstance(obj, (tuple, list)):
                for item in obj:
                    extract_refs(item)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_refs(value)

        extract_refs(op.args)
        extract_refs(op.kwargs)

        for src in input_refs:
            if src in self.graph:
                self.graph.add_edge(src, name)

        return name

    # ------------------------------------------------------------------- queries
    def nodes(self, *, sort: bool = True) -> list[str]:
        """获取所有节点名称

        Args:
            sort: 是否按创建顺序排序（默认 True）

        Returns:
            节点名称列表
        """
        if not sort:
            return list(self.graph.nodes)
        return sorted(self.graph.nodes, key=lambda n: self.graph.nodes[n].get("order", 0))

    def node_record(self, name: str) -> Op:
        """获取节点的 Op 对象

        Args:
            name: 节点名称

        Returns:
            节点的 Op 对象

        Raises:
            KeyError: 当节点不存在时
        """
        if name not in self.graph:
            raise KeyError(f"Node '{name}' not found")
        return self.graph.nodes[name]["op"]

    def successors(self, name: str) -> list[str]:
        """获取节点的直接后继节点

        返回当前节点输出直接连接的所有节点，即数据流方向的下游节点。
        对应 NetworkX 的 successors 方法。

        Args:
            name: 节点名称

        Returns:
            后继节点名称列表
        """
        return list(self.graph.successors(name))

    def predecessors(self, name: str) -> list[str]:
        """获取节点的直接前驱节点

        返回所有直接输出到当前节点的节点，即数据流方向的上游节点。
        用于分析节点的输入依赖。

        Args:
            name: 节点名称

        Returns:
            前驱节点名称列表
        """
        return list(self.graph.predecessors(name))

    # ---------------------------------------------------------------- mutations
    def remove_node(self, name: str, *, safe: bool = True) -> None:
        """移除节点

        从图中完全删除指定节点，包括相关的边连接。

        Args:
            name: 要移除的节点名称
            safe: 安全模式检查（默认 True）
                  - True: 如果节点有后继则抛出异常
                  - False: 强制删除，可能破坏图完整性

        Raises:
            RuntimeError: 安全模式下尝试移除有后继的节点
        """
        if name not in self.graph:
            return
        if safe and list(self.graph.successors(name)):
            raise RuntimeError(f"Cannot remove node '{name}' with existing users")
        self.graph.remove_node(name)

    def replace_input(self, node_name: str, old_input: str, new_input: str) -> None:
        """替换单个输入连接

        将指定节点的一个输入从旧节点改为新节点，同时更新：
        1. 边连接关系
        2. Op 对象的 args/kwargs 中的引用

        Args:
            node_name: 目标节点名称
            old_input: 要替换的输入节点名称
            new_input: 新的输入节点名称

        Raises:
            KeyError: 当任何指定节点不存在时
        """
        if node_name not in self.graph:
            raise KeyError(f"Node '{node_name}' not found")
        if old_input not in self.graph:
            raise KeyError(f"Input '{old_input}' not found")
        if new_input not in self.graph:
            raise KeyError(f"Replacement input '{new_input}' not found")

        # Update edges
        if self.graph.has_edge(old_input, node_name):
            self.graph.remove_edge(old_input, node_name)
        self.graph.add_edge(new_input, node_name)

        # Update Op's args and kwargs
        op = self.graph.nodes[node_name]["op"]
        op.args = self._replace_in_args(op.args, old_input, new_input)
        op.kwargs = self._replace_in_args(op.kwargs, old_input, new_input)

    def replace_uses(self, src: str, dst: str) -> None:
        """替换节点的所有使用

        将所有使用 src 节点的地方改为使用 dst 节点。

        操作包括：
        1. 将 src 的所有后继节点连接到 dst
        2. 删除 src 到这些后继的边
        3. 更新后继节点 Op 的 args/kwargs 中的引用

        Args:
            src: 源节点名称（被替换）
            dst: 目标节点名称（替换为）

        Raises:
            KeyError: 当指定节点不存在时
        """
        if src == dst:
            return
        if src not in self.graph or dst not in self.graph:
            raise KeyError("Both src and dst must exist in the graph")

        for succ in list(self.graph.successors(src)):
            # Update edge
            if not self.graph.has_edge(dst, succ):
                self.graph.add_edge(dst, succ)
            self.graph.remove_edge(src, succ)

            # Update Op's args and kwargs
            op = self.graph.nodes[succ]["op"]
            op.args = self._replace_in_args(op.args, src, dst)
            op.kwargs = self._replace_in_args(op.kwargs, src, dst)

    def merge_nodes(self, keep: str, remove: str) -> None:
        """合并两个节点

        将 remove 节点合并到 keep 节点，执行以下操作：
        1. 合并输入边：remove 的前驱连接到 keep
        2. 合并输出边：remove 的后继连接到 keep
        3. 合并元数据：remove 的 Op 元数据合并到 keep
        4. 删除 remove 节点

        用于节点优化和冗余消除。

        Args:
            keep: 要保留的节点名称
            remove: 要移除的节点名称

        Raises:
            KeyError: 当指定节点不存在时
        """
        if keep not in self.graph or remove not in self.graph:
            raise KeyError("Both nodes must exist to merge")

        # Merge input edges
        for pred in list(self.graph.predecessors(remove)):
            if pred == keep:
                continue
            if not self.graph.has_edge(pred, keep):
                self.graph.add_edge(pred, keep)

        # Merge output edges
        for succ in list(self.graph.successors(remove)):
            if succ == keep:
                continue
            if not self.graph.has_edge(keep, succ):
                self.graph.add_edge(keep, succ)

        # Merge metadata if exists
        keep_op = self.graph.nodes[keep]["op"]
        remove_op = self.graph.nodes[remove]["op"]
        if hasattr(remove_op, 'metadata') and isinstance(remove_op.metadata, dict):
            if hasattr(keep_op, 'metadata') and isinstance(keep_op.metadata, dict):
                keep_op.metadata.setdefault('custom', {}).update(remove_op.metadata.get('custom', {}))

        self.remove_node(remove, safe=False)

    def update_meta(self, name: str, **meta_updates: Any) -> None:
        """更新节点的元数据"""
        if name not in self.graph:
            raise KeyError(f"Node '{name}' not found")
        op = self.graph.nodes[name]["op"]
        if hasattr(op, 'metadata') and isinstance(op.metadata, dict):
            op.metadata.setdefault('custom', {}).update(meta_updates)

    def summarize(self) -> list[str]:
        """生成图的摘要信息"""
        lines: list[str] = []
        for name in self.nodes():
            op = self.graph.nodes[name]["op"]
            shape = op.results[0].shape if op.results else None
            dtype = op.results[0].dtype if op.results else None
            lines.append(f"{name}: {op.op_type}")

            if op.input_ops:
                lines.append("  inputs:")
                for input_op, rf in op.input_ops.items():
                    in_shape = input_op.results[rf.index].shape if input_op.results else None
                    in_dtype = input_op.results[rf.index].dtype if input_op.results else None
                    rf_str = f"[{rf.index}]"
                    if rf.transforms:
                        rf_str += f" -> {' -> '.join(t.op_type for t in rf.transforms)}"
                    lines.append(f"    - {input_op.op_type} {rf_str} {in_shape} {in_dtype}")

            if hasattr(op, 'kwargs') and op.kwargs:
                info_items = []
                for k, v in op.kwargs.items():
                    if v is not None and k not in ['transpose_a', 'transpose_b'] or v:
                        info_items.append(f"{k}={v}")
                if info_items:
                    lines.append(f"  params: {', '.join(info_items)}")

            lines.append(f"  output: {shape} {dtype}")
            successors = list(self.graph.successors(name))
            if successors:
                lines.append(f"  -> [{', '.join(successors)}]")

        return lines

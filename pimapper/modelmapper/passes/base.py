"""Pass 基类和 PassManager 实现

提供计算图优化 pass 的基础框架和管理器。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from pimapper.core.graph.base import NxComputationGraph


class Pass(ABC):
    """计算图优化 Pass 基类

    所有计算图优化 pass 都应该继承此基类并实现 run 方法。
    Pass 应该是无状态的，可以重复使用。

    Attributes:
        name: Pass 的名称，用于日志和调试
        description: Pass 的功能描述
    """

    def __init__(self, name: Optional[str] = None, description: Optional[str] = None):
        """初始化 Pass

        Args:
            name: Pass 名称（可选，默认使用类名）
            description: Pass 描述（可选）
        """
        self.name = name or self.__class__.__name__
        self.description = description or ""
        self._metadata: dict[str, Any] = {}

    @abstractmethod
    def run(self, graph: NxComputationGraph) -> bool:
        """执行 Pass 优化

        Args:
            graph: 要优化的计算图

        Returns:
            True 如果图被修改，False 如果图未改变
        """
        pass

    def get_metadata(self) -> dict[str, Any]:
        """获取 Pass 执行后的元数据

        Returns:
            元数据字典，包含 pass 执行的统计信息等
        """
        return dict(self._metadata)

    def reset_metadata(self) -> None:
        """重置元数据"""
        self._metadata.clear()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class PassManager:
    """Pass 管理器

    负责管理和执行一系列 pass，支持：
    - 顺序执行多个 pass
    - 迭代执行直到收敛
    - 收集执行统计信息
    - 提供执行日志

    Attributes:
        passes: 要执行的 pass 列表
        max_iterations: 最大迭代次数（用于迭代执行模式）
        verbose: 是否输出详细日志
    """

    def __init__(
        self,
        passes: Optional[list[Pass]] = None,
        *,
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        """初始化 PassManager

        Args:
            passes: Pass 列表（可选）
            max_iterations: 最大迭代次数
            verbose: 是否输出详细日志
        """
        self.passes: list[Pass] = passes or []
        self.max_iterations = max_iterations
        self.verbose = verbose
        self._stats: dict[str, Any] = {}

    def add_pass(self, pass_obj: Pass) -> PassManager:
        """添加一个 pass

        Args:
            pass_obj: 要添加的 pass

        Returns:
            self，支持链式调用
        """
        self.passes.append(pass_obj)
        return self

    def run(self, graph: NxComputationGraph, *, mode: str = "sequential") -> dict[str, Any]:
        """执行所有 pass

        Args:
            graph: 要优化的计算图
            mode: 执行模式
                - "sequential": 顺序执行所有 pass 一次
                - "iterative": 迭代执行直到图不再变化或达到最大迭代次数

        Returns:
            统计信息字典，包含：
            - modified: 图是否被修改
            - iterations: 迭代次数
            - pass_executions: 各个 pass 的执行次数和结果
        """
        if mode == "sequential":
            return self._run_sequential(graph)
        elif mode == "iterative":
            return self._run_iterative(graph)
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'sequential' or 'iterative'")

    def _run_sequential(self, graph: NxComputationGraph) -> dict[str, Any]:
        """顺序执行所有 pass 一次

        Args:
            graph: 要优化的计算图

        Returns:
            统计信息字典
        """
        stats: dict[str, Any] = {
            "modified": False,
            "iterations": 1,
            "pass_executions": {},
        }

        for pass_obj in self.passes:
            if self.verbose:
                print(f"[PassManager] Running {pass_obj.name}...")

            pass_obj.reset_metadata()
            modified = pass_obj.run(graph)

            stats["pass_executions"][pass_obj.name] = {
                "modified": modified,
                "metadata": pass_obj.get_metadata(),
            }

            if modified:
                stats["modified"] = True

            if self.verbose:
                status = "modified" if modified else "unchanged"
                print(f"[PassManager] {pass_obj.name} finished: {status}")

        self._stats = stats
        return stats

    def _run_iterative(self, graph: NxComputationGraph) -> dict[str, Any]:
        """迭代执行 pass 直到收敛

        Args:
            graph: 要优化的计算图

        Returns:
            统计信息字典
        """
        stats: dict[str, Any] = {
            "modified": False,
            "iterations": 0,
            "pass_executions": {},
        }

        for iteration in range(self.max_iterations):
            stats["iterations"] = iteration + 1
            iteration_modified = False

            if self.verbose:
                print(f"[PassManager] Iteration {iteration + 1}/{self.max_iterations}")

            for pass_obj in self.passes:
                if self.verbose:
                    print(f"[PassManager]   Running {pass_obj.name}...")

                pass_obj.reset_metadata()
                modified = pass_obj.run(graph)

                pass_key = f"{pass_obj.name}_iter_{iteration + 1}"
                stats["pass_executions"][pass_key] = {
                    "modified": modified,
                    "metadata": pass_obj.get_metadata(),
                }

                if modified:
                    iteration_modified = True
                    stats["modified"] = True

                if self.verbose:
                    status = "modified" if modified else "unchanged"
                    print(f"[PassManager]   {pass_obj.name} finished: {status}")

            # 如果本轮迭代没有任何修改，说明已收敛
            if not iteration_modified:
                if self.verbose:
                    print(f"[PassManager] Converged after {iteration + 1} iterations")
                break

        self._stats = stats
        return stats

    def get_stats(self) -> dict[str, Any]:
        """获取最后一次执行的统计信息

        Returns:
            统计信息字典
        """
        return dict(self._stats)

    def clear_passes(self) -> None:
        """清空所有 pass"""
        self.passes.clear()

    def __repr__(self) -> str:
        return f"PassManager(passes={len(self.passes)}, max_iterations={self.max_iterations})"

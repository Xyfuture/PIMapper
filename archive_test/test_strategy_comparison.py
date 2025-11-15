#!/usr/bin/env python3
"""
原始 Prompt:
请你参考下面的代码写一个针对 h2llm 和 recursive 这两个算法的测试文件,
测试几个矩阵的性能, 矩阵在代码中直接指定, 不需要读取文件.
同时要输出一下绝对的latency

Strategy Comparison Test: H2-LLM vs Recursive Grid Search

比较 H2-LLM 和 Recursive Grid Search 两种映射策略在不同矩阵上的性能表现。
"""

import logging
from typing import List, Tuple

from pimapper.core.hwspec import PIMChannelSpec, AcceleratorSpec, Accelerator
from pimapper.core.matrixspec import MatrixShape, DataFormat, DataType
from pimapper.matrixmapper.strategy.h2llm_mapping import H2LLMTilingStrategy
from pimapper.matrixmapper.strategy.recursive_grid_search import RecursiveGridSearchStrategy

# 禁用详细日志
logging.getLogger("h2llm_mapping").disabled = True
logging.getLogger("recursive_grid_search").disabled = True


def create_test_accelerator(channel_count: int = 8) -> Accelerator:
    """创建测试用的加速器配置"""
    channel_spec = PIMChannelSpec(
        compute_power=4,  # TFLOPS
        shared_bandwidth=12.5,  # GB/s
        memory_bandwidth=1.0  # TB/s
    )
    accel_spec = AcceleratorSpec(
        channel_count=channel_count,
        channel_spec=channel_spec
    )
    return Accelerator.create_from_spec(accel_spec)


def main():
    print("\n" + "="*100)
    print("Strategy Performance Comparison: H2-LLM vs Recursive Grid Search")
    print("="*100)

    # 创建加速器
    accelerator = create_test_accelerator(channel_count=8)
    print(f"Hardware: {len(accelerator.channels)} PIM channels")
    print(f"  Compute per channel: {accelerator.spec.channel_spec.compute_power} TFLOPS")
    print(f"  Bandwidth per channel: {accelerator.spec.channel_spec.shared_bandwidth} GB/s")
    print(f"  Total compute: {accelerator.total_compute_power} TFLOPS")
    print("="*100)

    # 定义测试矩阵 (rows, cols, batch_size)
    test_matrices: List[Tuple[str, int, int, int]] = [
        ("4096x4096", 4096, 4096, 20),
        ("4096x12288", 4096, 12288, 20),
        ("4096x14336", 4096, 14336, 20),
    ]

    # 数据格式: FP16 输入/输出, INT4 权重
    data_format = DataFormat(
        input_dtype=DataType.FP16,
        output_dtype=DataType.FP16,
        weight_dtype=DataType.INT4
    )

    # 创建策略
    h2llm_strategy = H2LLMTilingStrategy(element_size=2.0)
    recursive_strategy = RecursiveGridSearchStrategy(
        num_split_row_candidates=list(range(1, 12)),
        num_split_col_candidates=list(range(1, 12)),
        max_iterations=2,
    )

    # 测试每个矩阵
    for matrix_name, rows, cols, batch_size in test_matrices:
        print(f"\n{'='*100}")
        print(f"Matrix: {matrix_name} ({rows}x{cols}x{batch_size})")
        print(f"{'='*100}")

        # 创建矩阵
        matrix = MatrixShape(
            rows=rows,
            cols=cols,
            batch_size=batch_size,
            data_format=data_format
        )

        # 测试 H2-LLM 策略
        h2llm_result = h2llm_strategy.find_optimal_mapping(matrix, accelerator)
        if h2llm_result:
            print(f"\nH2-LLM Strategy:")
            print(f"  Latency: {h2llm_result.latency:.0f} cycles")
            print(f"  Tile shapes:")
            for ch_id, tiles in h2llm_result.mapping.placement.items():
                for tile in tiles:
                    print(f"    Channel {ch_id}: {tile.num_rows}x{tile.num_cols}x{tile.num_batches}")
                    break  # 只打印第一个tile（同一通道的tile形状相同）

        # 测试 Recursive 策略
        recursive_result = recursive_strategy.find_optimal_mapping(matrix, accelerator)
        if recursive_result:
            print(f"\nRecursive Strategy:")
            print(f"  Latency: {recursive_result.latency:.0f} cycles")
            print(f"  Tile shapes:")
            for ch_id, tiles in recursive_result.mapping.placement.items():
                for tile in tiles:
                    print(f"    Channel {ch_id}: {tile.num_rows}x{tile.num_cols}x{tile.num_batches}")
                    break  # 只打印第一个tile

        # 计算加速比
        if h2llm_result and recursive_result:
            speedup = h2llm_result.latency / recursive_result.latency
            print(f"\n  Speedup (H2-LLM/Recursive): {speedup:.2f}x")

    print("="*100)
    print("\nNote: Speedup = H2-LLM Latency / Recursive Latency (>1 means Recursive is faster)")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()

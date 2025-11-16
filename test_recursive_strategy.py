#!/usr/bin/env python3
"""
原始 Prompt:
参照 ref_recursive.py 中的代码, 看一下 recursive_grid_search.py 中的代码能不能满足原始的任务描述,
实现一个递归的算法, 让 recursive_grid_search 中递归调用 find_optimal_mapping, 同时保持好 matrix allocation tree 的构建.
编写测试文件只运行 recursive 策略, 启用log详细信息输出, 查看递归过程和最终结果.

Test Recursive Grid Search Strategy
测试递归网格搜索策略的详细递归过程和tile分配结果
"""

import logging

from pimapper.core.hwspec import PIMChannelSpec, AcceleratorSpec, Accelerator
from pimapper.core.matrixspec import MatrixShape, DataFormat, DataType
from pimapper.matrixmapper.strategy.recursive_grid_search import RecursiveGridSearchStrategy

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(message)s'
)

def main():
    print("=" * 100)
    print("递归网格搜索策略测试")
    print("=" * 100)

    # 创建加速器
    channel_spec = PIMChannelSpec(
        compute_power=4.0,
        shared_bandwidth=12.5,
        memory_bandwidth=0.4
    )
    accel_spec = AcceleratorSpec(
        channel_count=5,
        channel_spec=channel_spec
    )
    accelerator = Accelerator.create_from_spec(accel_spec)

    print(f"\n硬件配置:")
    print(f"  PIM Channel 数量: {len(accelerator.channels)}")
    print(f"  每个 Channel 的计算能力: {accelerator.spec.channel_spec.compute_power} TFLOPS")
    print(f"  总计算能力: {accelerator.total_compute_power} TFLOPS")

    # 创建矩阵
    data_format = DataFormat(
        input_dtype=DataType.FP16,
        output_dtype=DataType.FP16,
        weight_dtype=DataType.INT4
    )
    matrix = MatrixShape(
        rows=4096,
        cols=4096,
        batch_size=20,
        data_format=data_format
    )

    print(f"\n矩阵配置:")
    print(f"  维度: {matrix.rows}×{matrix.cols}×{matrix.batch_size}")
    print(f"  总元素数: {matrix.rows * matrix.cols * matrix.batch_size:,}")

    print("\n" + "=" * 100)
    print("开始递归网格搜索...")
    print("=" * 100 + "\n")

    # 创建策略
    strategy = RecursiveGridSearchStrategy(
        num_split_row_candidates=list(range(1, 5)),
        num_split_col_candidates=list(range(1, 5)),
        max_iterations=1,
    )

    # 执行搜索
    result = strategy.find_optimal_mapping(matrix, accelerator)

    # 打印结果
    print("\n" + "=" * 100)
    print("搜索完成")
    print("=" * 100)

    if not result:
        print("\n映射失败")
        return

    print(f"\n总延迟: {result.latency:.2f} cycles")
    print(f"计算利用率: {result.get_compute_utilization():.2%}")

    print(f"\n详细的 Tile 分配:")
    print("-" * 100)

    total_tiles = 0
    for ch_id in sorted(result.mapping.placement.keys()):
        tiles = result.mapping.placement[ch_id]
        total_tiles += len(tiles)
        print(f"\nChannel {ch_id} (共 {len(tiles)} 个 tiles):")
        for idx, tile in enumerate(tiles):
            print(f"  Tile {idx}: "
                  f"shape=({tile.num_rows}×{tile.num_cols}×{tile.num_batches}), "
                  f"size={tile.num_rows * tile.num_cols * tile.num_batches:,} elements")

    print(f"\n{'-' * 100}")
    print(f"统计信息:")
    print(f"  总 Tile 数量: {total_tiles}")
    print(f"  使用的 Channel 数量: {len(result.mapping.placement)}")
    print(f"  平均每个 Channel 的 Tile 数: {total_tiles / len(result.mapping.placement):.2f}")

    # 打印 allocation tree 信息
    if result.allocation_tree:
        print(f"\n分配树信息:")
        print(f"  根节点 split: {result.allocation_tree.root.num_split_row}×{result.allocation_tree.root.num_split_col}")
        print(f"  树深度: {result.allocation_tree.get_depth()}")
        print(f"  总节点数: {result.allocation_tree.count_nodes()}")

        if result.allocation_tree.validate(check_allocations=True):
            print(f"  树验证: ✓ 通过")
        else:
            print(f"  树验证: ✗ 失败")

    print("\n" + "=" * 100)
    print("测试完成")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()

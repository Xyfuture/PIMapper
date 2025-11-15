#!/usr/bin/env python3
"""
原始 Prompt:
请你参照 @archive_test\test_strategy_comparison.py 这个文件, 编写一个测试文件,
用于测试不同的 strategy 对于单个矩阵的性能. 这个文件中要同时初始化 3个 strategy,
同时设定一个参数, 决定可以启用那个strategy, accelerator spec 和 matrix 也都是可以指定的.
每个strategy的输出信息一定要详细, 要输出, 每个 matrix 都给具体分成了什么tile ,放到了 那个 channel 中

Single Matrix Strategy Detailed Test

测试单个矩阵在不同映射策略下的详细性能表现，包括完整的tile分配信息。
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass

from pimapper.core.hwspec import PIMChannelSpec, AcceleratorSpec, Accelerator
from pimapper.core.matrixspec import MatrixShape, DataFormat, DataType, Mapping
from pimapper.matrixmapper.strategy.h2llm_mapping import H2LLMTilingStrategy
from pimapper.matrixmapper.strategy.recursive_grid_search import RecursiveGridSearchStrategy
from pimapper.matrixmapper.strategy.trivial import TrivialTilingStrategy


@dataclass
class TestConfig:
    """测试配置"""
    # 启用的策略
    enable_trivial: bool = True
    enable_h2llm: bool = True
    enable_recursive: bool = True

    # 加速器配置
    channel_count: int = 8
    compute_power: float = 4.0  # TFLOPS
    shared_bandwidth: float = 12.8  # GB/s
    memory_bandwidth: float = 0.4  # TB/s

    # 矩阵配置
    matrix_rows: int = 4096
    matrix_cols: int = 4096
    matrix_batch_size: int = 20

    # 数据格式
    input_dtype: DataType = DataType.FP16
    output_dtype: DataType = DataType.FP16
    weight_dtype: DataType = DataType.INT4

    # 策略参数
    h2llm_element_size: float = 2.0
    recursive_max_iterations: int = 2
    recursive_row_candidates_max: int = 12
    recursive_col_candidates_max: int = 12
    trivial_grid_rows: int = 1
    # trivial_grid_cols: int = 8

    # 日志控制
    verbose_logging: bool = False

    @property
    def trivial_grid_cols(self):
        return self.channel_count


def create_accelerator(config: TestConfig) -> Accelerator:
    """根据配置创建加速器"""
    channel_spec = PIMChannelSpec(
        compute_power=config.compute_power,
        shared_bandwidth=config.shared_bandwidth,
        memory_bandwidth=config.memory_bandwidth
    )
    accel_spec = AcceleratorSpec(
        channel_count=config.channel_count,
        channel_spec=channel_spec
    )
    return Accelerator.create_from_spec(accel_spec)


def create_matrix(config: TestConfig) -> MatrixShape:
    """根据配置创建矩阵"""
    data_format = DataFormat(
        input_dtype=config.input_dtype,
        output_dtype=config.output_dtype,
        weight_dtype=config.weight_dtype
    )
    return MatrixShape(
        rows=config.matrix_rows,
        cols=config.matrix_cols,
        batch_size=config.matrix_batch_size,
        data_format=data_format
    )


def print_detailed_mapping(strategy_name: str, result, accelerator: Accelerator):
    """打印详细的映射信息"""
    if not result:
        print(f"\n{strategy_name}: 映射失败")
        return

    print(f"\n{'='*100}")
    print(f"{strategy_name} 策略结果")
    print(f"{'='*100}")
    print(f"总延迟: {result.latency:.2f} cycles")
    print(f"\n详细的 Tile 分配:")
    print(f"{'-'*100}")

    # 统计信息
    total_tiles = 0
    channel_tile_counts = {}

    # 遍历每个 channel 的 tile 分配
    for ch_id in sorted(result.mapping.placement.keys()):
        tiles = result.mapping.placement[ch_id]
        channel_tile_counts[ch_id] = len(tiles)
        total_tiles += len(tiles)

        print(f"\nChannel {ch_id} (共 {len(tiles)} 个 tiles):")
        for idx, tile in enumerate(tiles):
            print(f"  Tile {idx}: "
                  f"shape=({tile.num_rows}×{tile.num_cols}×{tile.num_batches}), "
                  f"size={tile.num_rows * tile.num_cols * tile.num_batches:,} elements")

    print(f"\n{'-'*100}")
    print(f"统计信息:")
    print(f"  总 Tile 数量: {total_tiles}")
    print(f"  使用的 Channel 数量: {len(result.mapping.placement)}")
    print(f"  平均每个 Channel 的 Tile 数: {total_tiles / len(result.mapping.placement):.2f}")

    # 打印每个 channel 的负载分布
    print(f"\nChannel 负载分布:")
    for ch_id in sorted(channel_tile_counts.keys()):
        count = channel_tile_counts[ch_id]
        percentage = (count / total_tiles) * 100
        bar_length = int(percentage / 2)  # 每2%一个字符
        bar = '█' * bar_length
        print(f"  Channel {ch_id}: {count:3d} tiles ({percentage:5.1f}%) {bar}")

    print(f"{'='*100}")


def main():
    # 创建测试配置
    config = TestConfig(
        # 启用的策略
        enable_trivial=True,
        enable_h2llm=True,
        enable_recursive=True,

        # 加速器配置
        channel_count=5,
        compute_power=4.0,
        shared_bandwidth=12.5,
        memory_bandwidth=0.4,

        # 矩阵配置
        matrix_rows=4096,
        matrix_cols=4096,
        matrix_batch_size=20,

        # 策略参数
        recursive_max_iterations=2,
        trivial_grid_rows=1,
        # trivial_grid_cols=8,

        # 日志控制
        verbose_logging=True
    )

    # 配置日志
    if not config.verbose_logging:
        logging.getLogger("h2llm_mapping").disabled = True
        logging.getLogger("recursive_grid_search").disabled = True
        logging.getLogger("trivial").disabled = True

    # 打印测试配置
    print("\n" + "="*100)
    print("单矩阵多策略详细性能测试")
    print("="*100)

    # 创建加速器
    accelerator = create_accelerator(config)
    print(f"\n硬件配置:")
    print(f"  PIM Channel 数量: {len(accelerator.channels)}")
    print(f"  每个 Channel 的计算能力: {accelerator.spec.channel_spec.compute_power} TFLOPS")
    print(f"  每个 Channel 的带宽: {accelerator.spec.channel_spec.shared_bandwidth} GB/s")
    print(f"  每个 Channel 的内存带宽: {accelerator.spec.channel_spec.memory_bandwidth} TB/s")
    print(f"  总计算能力: {accelerator.total_compute_power} TFLOPS")

    # 创建矩阵
    matrix = create_matrix(config)
    print(f"\n矩阵配置:")
    print(f"  维度: {matrix.rows}×{matrix.cols}×{matrix.batch_size}")
    print(f"  总元素数: {matrix.rows * matrix.cols * matrix.batch_size:,}")
    print(f"  数据格式: input={config.input_dtype.name}, "
          f"output={config.output_dtype.name}, weight={config.weight_dtype.name}")

    print(f"\n启用的策略:")
    if config.enable_trivial:
        print(f"  [*] Trivial ({config.trivial_grid_rows}x{config.trivial_grid_cols} 划分)")
    if config.enable_h2llm:
        print(f"  [*] H2-LLM ")
    if config.enable_recursive:
        print(f"  [*] Recursive Grid Search (max_iterations={config.recursive_max_iterations})")

    print("="*100)

    # 存储结果用于比较
    results = {}

    # 测试 Trivial 策略
    if config.enable_trivial:
        print("\n\n正在测试 Trivial 策略...")
        trivial_strategy = TrivialTilingStrategy()
        trivial_result = trivial_strategy.find_optimal_mapping(
            matrix, accelerator, grid_rows=config.trivial_grid_rows, grid_cols=config.trivial_grid_cols
        )
        print_detailed_mapping(f"Trivial ({config.trivial_grid_rows}x{config.trivial_grid_cols})", trivial_result, accelerator)
        results['Trivial'] = trivial_result

    # 测试 H2-LLM 策略
    if config.enable_h2llm:
        print("\n\n正在测试 H2-LLM 策略...")
        h2llm_strategy = H2LLMTilingStrategy()
        h2llm_result = h2llm_strategy.find_optimal_mapping(matrix, accelerator)
        print_detailed_mapping("H2-LLM", h2llm_result, accelerator)
        results['H2-LLM'] = h2llm_result

    # 测试 Recursive 策略
    if config.enable_recursive:
        print("\n\n正在测试 Recursive Grid Search 策略...")
        recursive_strategy = RecursiveGridSearchStrategy(
            num_split_row_candidates=list(range(1, config.recursive_row_candidates_max)),
            num_split_col_candidates=list(range(1, config.recursive_col_candidates_max)),
            max_iterations=config.recursive_max_iterations,
        )
        recursive_result = recursive_strategy.find_optimal_mapping(matrix, accelerator)
        print_detailed_mapping("Recursive Grid Search", recursive_result, accelerator)
        results['Recursive'] = recursive_result

    # 性能比较
    if len(results) > 1:
        print("\n\n" + "="*100)
        print("性能比较")
        print("="*100)

        # 找出最佳策略（过滤掉 latency 为 0 或 None 的结果）
        valid_results = {name: result for name, result in results.items()
                        if result is not None and result.latency > 0}

        if valid_results:
            best_strategy = min(valid_results.items(), key=lambda x: x[1].latency)

            print(f"\n延迟对比:")
            for name in sorted(valid_results.keys()):
                result = valid_results[name]
                speedup = result.latency / best_strategy[1].latency
                is_best = " [最佳]" if name == best_strategy[0] else ""
                print(f"  {name:25s}: {result.latency:12.2f} cycles  "
                      f"(相对最佳: {speedup:.2f}x){is_best}")

            print(f"\n最佳策略: {best_strategy[0]} (延迟: {best_strategy[1].latency:.2f} cycles)")
        else:
            print("\n没有有效的延迟数据可供比较")

        # 显示所有结果（包括 latency 为 0 的）
        print(f"\n所有策略结果:")
        for name in sorted(results.keys()):
            result = results[name]
            if result is None:
                print(f"  {name:25s}: 映射失败")
            elif result.latency == 0:
                print(f"  {name:25s}: 映射成功 (未计算延迟)")
            else:
                print(f"  {name:25s}: {result.latency:12.2f} cycles")

        print("="*100)

    print("\n测试完成!\n")


if __name__ == "__main__":
    main()

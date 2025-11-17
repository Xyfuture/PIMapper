"""Benchmark strategies for edge device configuration.

Original prompt: 创建 bench_strategy_edge.py 用于边缘设备配置的策略测试
"""

import multiprocessing
from pimapper.utils.benchmark import setup_logging, setup_csv_output, run_single_test, log_latency_details, finalize_csv
from pimapper.core.hwspec import PIMChannelSpec, AcceleratorSpec, HostSpec
from pimapper.core.matrixspec import DataFormat, DataType


# Edge device hardware configuration
NUMBER_OF_CHANNELS = 8

CHANNEL_SPEC = PIMChannelSpec(
    compute_power=4,           # 2 TOPS (edge device)
    shared_bandwidth=12.8,        # 6.4 GB/s
    memory_bandwidth=0.4,        # 200 GB/s
)

HOST_SPEC = HostSpec(
    vector_compute_power=128      # 64 GFLOPS
)

ACCELERATOR_SPEC = AcceleratorSpec(
    channel_count=NUMBER_OF_CHANNELS,
    channel_spec=CHANNEL_SPEC,
    host_spec=HOST_SPEC
)

# Inference configuration
PAST_SEQ_LEN = 1024
DATA_FORMAT = DataFormat(
    input_dtype=DataType.FP16,
    output_dtype=DataType.FP16,
    weight_dtype=DataType.INT4
)

# Test matrix
STRATEGIES = {
    "h2llm": {},
    "recursive_grid_search": {
        "num_split_row_candidates": list(range(1, 9)),
        "num_split_col_candidates": list(range(1, 9)),
        "max_iterations": 1,
    }
}

MODELS = [ "Qwen3-4B","Qwen3-8B","Qwen3-14B"]
BATCH_SIZES = [1, 4, 8]


def run_benchmark():
    """Run edge device benchmark."""
    logger, log_file = setup_logging()
    logger.info("="*80)
    logger.info("Edge Device Strategy Benchmark")
    logger.info("="*80)
    logger.info(f"Log file: {log_file}")

    tmp_csv_file, final_csv_file, csv_writer, csv_handle = setup_csv_output()
    logger.info(f"CSV file: {final_csv_file}")
    logger.info("")

    logger.info("Configuration:")
    logger.info(f"  Hardware: {ACCELERATOR_SPEC.channel_count} channels, {CHANNEL_SPEC.compute_power} TOPS")
    logger.info(f"  Inference: past_seq_len={PAST_SEQ_LEN}, data_format=FP16")
    logger.info(f"  Strategies: {list(STRATEGIES.keys())}")
    logger.info(f"  Models: {MODELS}")
    logger.info(f"  Batch sizes: {BATCH_SIZES}")
    logger.info("")

    test_configs = [
        (model, batch, strategy, config, ACCELERATOR_SPEC, PAST_SEQ_LEN, DATA_FORMAT)
        for model in MODELS
        for batch in BATCH_SIZES
        for strategy, config in STRATEGIES.items()
    ]

    total_tests = len(test_configs)
    logger.info(f"Total tests: {total_tests}")
    logger.info("Starting parallel execution...")
    logger.info("")

    results = []
    pool = multiprocessing.Pool(processes=4)
    try:
        async_results = [pool.apply_async(run_single_test, (config,)) for config in test_configs]

        for i, async_result in enumerate(async_results, 1):
            try:
                result = async_result.get()
            except Exception as e:
                model, batch, strategy = test_configs[i-1][:3]
                result = {
                    'model_name': model, 'batch_size': batch, 'strategy': strategy,
                    'total_latency': -1, 'matrix_ops_count': -1, 'vector_ops_count': -1,
                    'success': False, 'error': f'Process crashed: {str(e)}'
                }

            results.append(result)
            csv_writer.writerow(result)
            csv_handle.flush()

            status = "✓" if result['success'] else "✗"
            logger.info(f"[{i}/{total_tests}] {status} {result['model_name']} "
                       f"batch={result['batch_size']} strategy={result['strategy']} "
                       f"latency={result['total_latency']:.6f}s")

            if not result['success']:
                logger.error(f"  Error: {result['error']}")
            else:
                log_latency_details(logger, result)
    finally:
        pool.close()
        pool.terminate()
        pool.join()
        csv_handle.close()

    # Sort and finalize CSV
    logger.info("\nFinalizing results...")
    finalize_csv(tmp_csv_file, final_csv_file, MODELS, BATCH_SIZES, list(STRATEGIES.keys()))

    logger.info("")
    logger.info("="*80)
    logger.info("Benchmark Completed")
    logger.info("="*80)
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Successful: {sum(1 for r in results if r['success'])}")
    logger.info(f"Failed: {sum(1 for r in results if not r['success'])}")
    logger.info(f"Results: {final_csv_file}")
    logger.info(f"Log: {log_file}")

    return results


if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn', force=True)
    run_benchmark()

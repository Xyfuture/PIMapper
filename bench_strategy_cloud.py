"""Benchmark strategies for cloud server configuration.

Original prompt: 创建 bench_strategy_cloud.py 用于云服务器配置的策略测试
"""

from multiprocessing import Pool
from pimapper.utils.benchmark import setup_logging, setup_csv_output, run_single_test, log_latency_details
from pimapper.core.hwspec import PIMChannelSpec, AcceleratorSpec, HostSpec
from pimapper.core.matrixspec import DataFormat, DataType


# Cloud server hardware configuration
NUMBER_OF_CHANNELS = 8

CHANNEL_SPEC = PIMChannelSpec(
    compute_power=16,           # 8 TOPS (cloud server)
    shared_bandwidth=20.8,       # 25.6 GB/s
    memory_bandwidth=1.6,        # 800 GB/s
)

HOST_SPEC = HostSpec(
    vector_compute_power=128*4     # 256 GFLOPS
)

ACCELERATOR_SPEC = AcceleratorSpec(
    channel_count=NUMBER_OF_CHANNELS,
    channel_spec=CHANNEL_SPEC,
    host_spec=HOST_SPEC
)

# Inference configuration
PAST_SEQ_LEN = 2048
DATA_FORMAT = DataFormat(
    input_dtype=DataType.FP16,
    output_dtype=DataType.FP16,
    weight_dtype=DataType.FP16
)

# Test matrix
STRATEGIES = {
    "h2llm": {},
    "recursive_grid_search": {
        "num_split_row_candidates": list(range(1, 9)),
        "num_split_col_candidates": list(range(1, 9)),
        "max_iterations": 2,
    }
}

MODELS = ["Qwen3-32B","Yi-34B","Meta-Llama-3-70B"]
BATCH_SIZES = [16, 32, 64]


def run_benchmark():
    """Run cloud server benchmark."""
    logger, log_file = setup_logging()
    logger.info("="*80)
    logger.info("Cloud Server Strategy Benchmark")
    logger.info("="*80)
    logger.info(f"Log file: {log_file}")

    csv_file, csv_writer, csv_handle = setup_csv_output()
    logger.info(f"CSV file: {csv_file}")
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
    with Pool(processes=8) as pool:
        for i, result in enumerate(pool.imap_unordered(run_single_test, test_configs), 1):
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

    csv_handle.close()

    logger.info("")
    logger.info("="*80)
    logger.info("Benchmark Completed")
    logger.info("="*80)
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Successful: {sum(1 for r in results if r['success'])}")
    logger.info(f"Failed: {sum(1 for r in results if not r['success'])}")
    logger.info(f"Results: {csv_file}")
    logger.info(f"Log: {log_file}")

    return results


if __name__ == "__main__":
    run_benchmark()

"""Test strategy benchmark for different models and batch sizes.

Original prompt:
请你编写一个大的测试文件, 用于帮我运算实验的结果, 将测试中的log 存储起来,
并将结果存储到一个csv文件中. 总的来说, 这个测试是为了获得, 不同 mapping strategy,
在不同 batch size, 不同模型下的, 延迟的结果.

Test Configuration:
- Hardware: 8 channels, 4 TOPS compute, 400 GB/s memory bandwidth, 12.8 GB/s I/O bandwidth
- Inference: past_seq_len=1024, data_format=FP16
- Strategies: h2llm, recursive_grid_search
- Models: Meta-Llama-3-8B, Qwen3-4B, Qwen3-8B, Qwen3-32B
- Batch sizes: [1, 4, 8, 16, 20]

Pass execution logic:
- All strategies: NormalizeOpsPass -> SimplifyGraphPass
- h2llm: MatrixMappingPass -> VectorLatencyPass -> LatencyCalculationPass
- recursive_grid_search: MatrixFusionPass -> MatrixMappingPass -> VectorLatencyPass -> LatencyCalculationPass

Output:
- Log file: logs/benchmark_YYYYMMDD_HHMMSS.log
- CSV file: results/benchmark_results_YYYYMMDD_HHMMSS.csv
"""

import torch
import logging
import csv
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from typing import Dict, List, Tuple, Any

from pimapper.model.base import load_model_config, initialize_module, InferenceConfig
from pimapper.modelmapper.converter import trace_module, fx_to_computation_graph
from pimapper.modelmapper.passes.normalize_ops import NormalizeOpsPass
from pimapper.modelmapper.passes.simplify import SimplifyGraphPass
from pimapper.modelmapper.passes.matrix_fusion import MatrixFusionPass
from pimapper.codegen.passes.matrix_mapping import MatrixMappingPass
from pimapper.codegen.passes.latency_calculation import LatencyCalculationPass
from pimapper.codegen.passes.vector_latency import VectorLatencyPass
from pimapper.core.hwspec import PIMChannelSpec, AcceleratorSpec, HostSpec
from pimapper.core.matrixspec import DataFormat, DataType


# ============================================================================
# Configuration
# ============================================================================

# Hardware configuration (fixed)
NUMBER_OF_CHANNELS = 8 

CHANNEL_SPEC = PIMChannelSpec(
    compute_power=4.0,           # 4 TOPS
    shared_bandwidth= 12.8,      # 12.8 GB/s 不要改动这个
    memory_bandwidth=0.4,        # 400 GB/s 
)

HOST_SPEC = HostSpec(
    vector_compute_power=128   # 128 GFLOPS for vector operations
)

ACCELERATOR_SPEC = AcceleratorSpec(
    channel_count=NUMBER_OF_CHANNELS,
    channel_spec=CHANNEL_SPEC,
    host_spec=HOST_SPEC
)

# Inference configuration (fixed except batch_size)
PAST_SEQ_LEN = 1024
DATA_FORMAT = DataFormat(
    input_dtype=DataType.FP16,
    output_dtype=DataType.FP16,
    weight_dtype=DataType.FP16
)

# Test matrix
STRATEGIES = {
    "h2llm": {
        # "element_size": 2.0  # FP16
    },
    "recursive_grid_search": {
        "num_split_row_candidates": list(range(1, 12)),
        "num_split_col_candidates": list(range(1, 12)),
        "max_iterations": 2,
    }
}

MODELS = [
    "Meta-Llama-3-8B",
    "Qwen3-4B",
    "Qwen3-8B",
    # "Qwen3-32B"
]

BATCH_SIZES = [16,32,64]


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging() -> Tuple[logging.Logger, Path]:
    """Setup logging to file and console."""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"benchmark_{timestamp}.log"

    # Configure logger
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


# ============================================================================
# Core Functions
# ============================================================================

def build_base_computation_graph(
    model_name: str,
    batch_size: int,
    logger: logging.Logger
):
    """Build base computation graph with common passes.

    This function:
    1. Loads model config and initializes LLaMA layer
    2. Traces the module to get fx_graph
    3. Converts to computation graph
    4. Runs NormalizeOpsPass
    5. Runs SimplifyGraphPass

    Args:
        model_name: Name of the model (e.g., "Meta-Llama-3-8B")
        batch_size: Batch size for inference
        logger: Logger instance

    Returns:
        Computation graph after common passes
    """
    logger.info(f"Building base computation graph for {model_name}, batch_size={batch_size}")

    # Load model config
    card_path = Path(f"pimapper/model/model_cards/{model_name}.json")
    if not card_path.exists():
        raise FileNotFoundError(f"Model card not found: {card_path}")

    config = load_model_config(card_path)
    logger.info(f"  Model config: hidden_size={config.hidden_size}, "
                f"intermediate_size={config.intermediate_size}, "
                f"num_attention_heads={config.num_attention_heads}")

    # Create inference config
    inference_config = InferenceConfig(
        batch_size=batch_size,
        past_seq_len=PAST_SEQ_LEN,
        data_format=DATA_FORMAT
    )

    # Initialize module
    llama_layer = initialize_module(config, inference_config=inference_config, dtype=torch.float16)
    llama_layer.eval()

    # Trace module (seq_len is always 1 for current token)
    sample_input = torch.randn(batch_size, 1, config.hidden_size, dtype=torch.float16)
    logger.info(f"  Tracing module with input shape: {sample_input.shape}")

    fx_graph = trace_module(llama_layer, sample_inputs=(sample_input,), inference_config=inference_config)
    comp_graph = fx_to_computation_graph(fx_graph, llama_layer, inference_config=inference_config)

    logger.info(f"  Initial graph: {len(comp_graph.nodes())} nodes")

    # Run common passes
    logger.info("  Running NormalizeOpsPass...")
    NormalizeOpsPass().run(comp_graph)
    logger.info(f"    After normalization: {len(comp_graph.nodes())} nodes")

    logger.info("  Running SimplifyGraphPass...")
    SimplifyGraphPass().run(comp_graph)
    logger.info(f"    After simplification: {len(comp_graph.nodes())} nodes")

    return comp_graph


def run_strategy_passes(
    graph,
    strategy_name: str,
    strategy_config: Dict[str, Any],
    accelerator_spec: AcceleratorSpec,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Run strategy-specific passes and calculate latency.

    Args:
        graph: Computation graph (will be modified in-place)
        strategy_name: Name of the strategy ("h2llm" or "recursive_grid_search")
        strategy_config: Strategy configuration parameters
        accelerator_spec: Hardware accelerator specification
        logger: Logger instance

    Returns:
        Dictionary containing latency metadata
    """
    logger.info(f"  Running passes for strategy: {strategy_name}")

    # Strategy-specific passes
    if strategy_name == "recursive_grid_search":
        # Run matrix fusion pass for recursive_grid_search
        logger.info("    Running MatrixFusionPass...")
        fusion_pass = MatrixFusionPass(min_fusion_size=2, block_size=64)
        fusion_pass.run(graph)
        logger.info(f"      After fusion: {len(graph.nodes())} nodes")

    # Run matrix mapping pass (all strategies)
    logger.info(f"    Running MatrixMappingPass with strategy={strategy_name}...")
    mapping_pass = MatrixMappingPass(
        accelerator_spec=accelerator_spec,
        strategy=strategy_name,
        strategy_kwargs=strategy_config,
    )
    mapping_pass.run(graph)


    # Run vector latency pass (exclude recursive_grid_search )
    if strategy_name != 'recursive_grid_search':
        logger.info("    Running VectorLatencyPass...")
        vector_pass = VectorLatencyPass(accelerator_spec.host_spec)
        vector_pass.run(graph)

    # Run latency calculation pass (all strategies)
    logger.info("    Running LatencyCalculationPass...")
    latency_pass = LatencyCalculationPass()
    latency_pass.run(graph)

    # Get metadata
    metadata = latency_pass.get_metadata()
    logger.info(f"    Total latency: {metadata['total_latency']:.6f} seconds")
    logger.info(f"    Matrix ops: {metadata['matrix_ops_count']}, "
                f"Vector ops: {metadata['vector_ops_count']}")

    return metadata


# ============================================================================
# CSV Output
# ============================================================================

def setup_csv_output() -> Tuple[Path, csv.DictWriter]:
    """Setup CSV output file and writer."""
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Create CSV file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = results_dir / f"benchmark_results_{timestamp}.csv"

    # Open file and create writer
    file_handle = open(csv_file, 'w', newline='', encoding='utf-8')
    fieldnames = [
        'model_name',
        'batch_size',
        'strategy',
        'total_latency',
        'matrix_ops_count',
        'vector_ops_count',
        'timestamp'
    ]
    writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
    writer.writeheader()

    return csv_file, writer, file_handle


# ============================================================================
# Main Test Loop
# ============================================================================

def run_benchmark():
    """Run the complete benchmark test."""
    # Setup logging
    logger, log_file = setup_logging()
    logger.info("="*80)
    logger.info("Starting Strategy Benchmark Test")
    logger.info("="*80)
    logger.info(f"Log file: {log_file}")

    # Setup CSV output
    csv_file, csv_writer, csv_handle = setup_csv_output()
    logger.info(f"CSV file: {csv_file}")
    logger.info("")

    # Print configuration
    logger.info("Configuration:")
    logger.info(f"  Hardware: {ACCELERATOR_SPEC.channel_count} channels, "
                f"{CHANNEL_SPEC.compute_power} TOPS compute")
    logger.info(f"  Inference: past_seq_len={PAST_SEQ_LEN}, data_format=FP16")
    logger.info(f"  Strategies: {list(STRATEGIES.keys())}")
    logger.info(f"  Models: {MODELS}")
    logger.info(f"  Batch sizes: {BATCH_SIZES}")
    logger.info("")

    # Results collection
    results = []
    total_tests = len(MODELS) * len(BATCH_SIZES) * len(STRATEGIES)
    current_test = 0

    # Main test loop
    for model_name in MODELS:
        logger.info("="*80)
        logger.info(f"Testing model: {model_name}")
        logger.info("="*80)

        for batch_size in BATCH_SIZES:
            logger.info(f"\nBatch size: {batch_size}")
            logger.info("-"*80)

            try:
                # Build base computation graph (only once per model/batch_size)
                base_graph = build_base_computation_graph(model_name, batch_size, logger)

                for strategy_name, strategy_config in STRATEGIES.items():
                    current_test += 1
                    logger.info(f"\n[Test {current_test}/{total_tests}] Strategy: {strategy_name}")

                    try:
                        # Deep copy graph for this strategy
                        graph_copy = deepcopy(base_graph)

                        # Run strategy-specific passes
                        metadata = run_strategy_passes(
                            graph_copy,
                            strategy_name,
                            strategy_config,
                            ACCELERATOR_SPEC,
                            logger
                        )

                        # Record result
                        result = {
                            'model_name': model_name,
                            'batch_size': batch_size,
                            'strategy': strategy_name,
                            'total_latency': metadata['total_latency'],
                            'matrix_ops_count': metadata['matrix_ops_count'],
                            'vector_ops_count': metadata['vector_ops_count'],
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        results.append(result)

                        # Write to CSV immediately
                        csv_writer.writerow(result)
                        csv_handle.flush()

                        logger.info(f"  ✓ Test completed successfully")

                    except Exception as e:
                        logger.error(f"  ✗ Error in strategy {strategy_name}: {str(e)}", exc_info=True)
                        # Record error result
                        result = {
                            'model_name': model_name,
                            'batch_size': batch_size,
                            'strategy': strategy_name,
                            'total_latency': -1,  # Error indicator
                            'matrix_ops_count': -1,
                            'vector_ops_count': -1,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        results.append(result)
                        csv_writer.writerow(result)
                        csv_handle.flush()

            except Exception as e:
                logger.error(f"Error building graph for {model_name}, batch_size={batch_size}: {str(e)}",
                           exc_info=True)
                # Skip all strategies for this model/batch_size combination
                for strategy_name in STRATEGIES.keys():
                    current_test += 1
                    result = {
                        'model_name': model_name,
                        'batch_size': batch_size,
                        'strategy': strategy_name,
                        'total_latency': -1,
                        'matrix_ops_count': -1,
                        'vector_ops_count': -1,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    results.append(result)
                    csv_writer.writerow(result)
                    csv_handle.flush()

    # Close CSV file
    csv_handle.close()

    # Summary
    logger.info("")
    logger.info("="*80)
    logger.info("Benchmark Test Completed")
    logger.info("="*80)
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Successful tests: {sum(1 for r in results if r['total_latency'] >= 0)}")
    logger.info(f"Failed tests: {sum(1 for r in results if r['total_latency'] < 0)}")
    logger.info(f"Results saved to: {csv_file}")
    logger.info(f"Log saved to: {log_file}")
    logger.info("")

    return results


if __name__ == "__main__":
    run_benchmark()

"""Benchmark utilities for testing mapping strategies.

Original prompt: 将 test_strategy_benchmark.py 的核心函数放到 pimapper/utils/ 下面
"""

import torch
import logging
import csv
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from pimapper.model.base import load_model_config, initialize_module, InferenceConfig
from pimapper.modelmapper.converter import trace_module, fx_to_computation_graph
from pimapper.modelmapper.passes.normalize_ops import NormalizeOpsPass
from pimapper.modelmapper.passes.simplify import SimplifyGraphPass
from pimapper.modelmapper.passes.matrix_fusion import MatrixFusionPass
from pimapper.codegen.passes.matrix_mapping import MatrixMappingPass
from pimapper.codegen.passes.latency_calculation import LatencyCalculationPass
from pimapper.codegen.passes.vector_latency import VectorLatencyPass
from pimapper.core.hwspec import AcceleratorSpec


def setup_logging(log_dir: str = "logs") -> tuple:
    """Setup logging to file and console."""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"benchmark_{timestamp}.log"

    logger = logging.getLogger(f"benchmark_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


def setup_csv_output(results_dir: str = "results") -> tuple:
    """Setup CSV output file and writer."""
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = results_path / f"benchmark_results_{timestamp}.csv"

    file_handle = open(csv_file, 'w', newline='', encoding='utf-8')
    fieldnames = [
        'model_name', 'batch_size', 'strategy', 'total_latency',
        'matrix_ops_count', 'vector_ops_count', 'timestamp',
        'success', 'error', 'traceback'
    ]
    writer = csv.DictWriter(file_handle, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()

    return csv_file, writer, file_handle


def run_single_test(args: tuple) -> Dict[str, Any]:
    """Run single test configuration in isolated process."""
    model_name, batch_size, strategy_name, strategy_config, accelerator_spec, past_seq_len, data_format = args

    try:
        card_path = Path(f"pimapper/model/model_cards/{model_name}.json")
        config = load_model_config(card_path)

        inference_config = InferenceConfig(
            batch_size=batch_size,
            past_seq_len=past_seq_len,
            data_format=data_format
        )

        llama_layer = initialize_module(config, inference_config=inference_config, dtype=torch.float16)
        llama_layer.eval()

        sample_input = torch.randn(batch_size, 1, config.hidden_size, dtype=torch.float16)
        fx_graph = trace_module(llama_layer, sample_inputs=(sample_input,), inference_config=inference_config)
        comp_graph = fx_to_computation_graph(fx_graph, llama_layer, inference_config=inference_config)

        NormalizeOpsPass().run(comp_graph)
        SimplifyGraphPass().run(comp_graph)

        if strategy_name == "recursive_grid_search":
            MatrixFusionPass(min_fusion_size=2, block_size=64).run(comp_graph)

        mapping_pass = MatrixMappingPass(
            accelerator_spec=accelerator_spec,
            strategy=strategy_name,
            strategy_kwargs=strategy_config,
        )
        mapping_pass.run(comp_graph)

        if strategy_name != 'recursive_grid_search':
            vector_pass = VectorLatencyPass(accelerator_spec.host_spec)
            vector_pass.run(comp_graph)

        latency_pass = LatencyCalculationPass()
        latency_pass.run(comp_graph)
        metadata = latency_pass.get_metadata()

        return {
            'model_name': model_name,
            'batch_size': batch_size,
            'strategy': strategy_name,
            'total_latency': metadata['total_latency'],
            'matrix_ops_count': metadata['matrix_ops_count'],
            'vector_ops_count': metadata['vector_ops_count'],
            'latency_details': metadata['latency_details'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'success': True
        }
    except Exception as e:
        return {
            'model_name': model_name,
            'batch_size': batch_size,
            'strategy': strategy_name,
            'total_latency': -1,
            'matrix_ops_count': -1,
            'vector_ops_count': -1,
            'latency_details': [],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def log_latency_details(logger: logging.Logger, result: Dict[str, Any]):
    """Log detailed latency information for each operation."""
    if not result['success'] or not result.get('latency_details'):
        return

    logger.info(f"\n{'='*80}")
    logger.info(f"Detailed Latency: {result['model_name']} batch={result['batch_size']} strategy={result['strategy']}")
    logger.info(f"{'='*80}")

    for detail in result['latency_details']:
        logger.info(f"\n  {detail['node_name']}")
        logger.info(f"    Op: {detail['op_type']}")
        if 'matrix_shape' in detail:
            logger.info(f"    Shape: {detail['matrix_shape']['rows']}x{detail['matrix_shape']['cols']} (batch={detail['matrix_shape']['batch_size']})")
        elif 'output_shape' in detail:
            logger.info(f"    Output Shape: {detail['output_shape']}")
        logger.info(f"    Latency: {detail['latency']:.6f}")
        logger.info(f"    Cumulative: {detail['cumulative_latency']:.6f}")

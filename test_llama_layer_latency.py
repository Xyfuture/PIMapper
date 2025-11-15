"""Test latency calculation pass with complete LLaMA layer.

Original prompt: 创建测试文件测试完整 LLaMA 层的延迟
- 硬件配置: 8 channels, 4 TOPS compute, 400 GB/s memory bandwidth, 12.8 GB/s I/O bandwidth
- 测试完整的 LLaMA transformer 层 (batch_size=20)
- 使用 matrix fusion pass
- 测试两种映射策略: h2llm 和 recursive_grid_search
"""

import torch
from pathlib import Path

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


def test_llama_layer_latency():
    # Load model config and create LLaMA layer
    card_path = Path("pimapper/model/model_cards/Meta-Llama-3-8B.json")
    config = load_model_config(card_path)

    # Create data format (FP16)
    data_format = DataFormat(
        input_dtype=DataType.FP16,
        output_dtype=DataType.FP16,
        weight_dtype=DataType.FP16
    )

    # Create inference config with data format
    inference_config = InferenceConfig(
        batch_size=20,
        past_seq_len=1024,
        data_format=data_format
    )

    llama_layer = initialize_module(config, inference_config=inference_config, dtype=torch.float16)
    llama_layer.eval()

    # Trace LLaMA layer, seq_len is always 1 for current token
    sample_input = torch.randn(inference_config.batch_size, 1, config.hidden_size, dtype=torch.float16)
    fx_graph = trace_module(llama_layer, sample_inputs=(sample_input,), inference_config=inference_config)
    comp_graph = fx_to_computation_graph(fx_graph, llama_layer, inference_config=inference_config)

    # Apply normalization and simplification passes
    NormalizeOpsPass().run(comp_graph)
    SimplifyGraphPass().run(comp_graph)

    # Apply matrix fusion pass
    MatrixFusionPass(min_fusion_size=2, block_size=64).run(comp_graph)

    # Hardware spec (matching test_strategy_benchmark.py configuration)
    channel_spec = PIMChannelSpec(
        compute_power=4.0,           # 4 TOPS
        shared_bandwidth=12.8,       # 12.8 GB/s (I/O bandwidth)
        memory_bandwidth=0.4,        # 400 GB/s (note: stored as 0.4, likely in TB/s)
    )
    host_spec = HostSpec(vector_compute_power=128.0)  # 128 GFLOPS for vector operations
    accelerator_spec = AcceleratorSpec(
        channel_count=8,
        channel_spec=channel_spec,
        host_spec=host_spec
    )

    # Test both strategies
    for strategy in ["h2llm", "recursive_grid_search"]:
        print(f"\n{'='*80}")
        print(f"Strategy: {strategy}")
        print(f"{'='*80}")

        strategy_kwargs = {}
        if strategy == 'recursive_grid_search':
            strategy_kwargs = {
                'num_split_row_candidates':list(range(1, 12)),
                'num_split_col_candidates':list(range(1, 12)),
                'max_iterations':2,
            }

        # Run matrix mapping pass
        mapping_pass = MatrixMappingPass(
            accelerator_spec=accelerator_spec,
            strategy=strategy,
            strategy_kwargs=strategy_kwargs,
        )
        mapping_pass.run(comp_graph)

        # Run vector latency pass
        vector_pass = VectorLatencyPass(accelerator_spec.host_spec)
        # vector_pass.run(comp_graph)

        # Run latency calculation pass
        latency_pass = LatencyCalculationPass()
        latency_pass.run(comp_graph)

        # Get metadata
        metadata = latency_pass.get_metadata()

        print(f"\nTotal latency: {metadata['total_latency']:.6f}")
        print(f"Matrix operations count: {metadata['matrix_ops_count']}")
        print(f"Vector operations count: {metadata['vector_ops_count']}")
        print(f"\nLatency Details (Topological Order):")

        for detail in metadata['latency_details']:
            print(f"\n  {detail['node_name']}")
            print(f"    Op: {detail['op_type']}")
            if 'matrix_shape' in detail:
                print(f"    Shape: {detail['matrix_shape']['rows']}x{detail['matrix_shape']['cols']} (batch={detail['matrix_shape']['batch_size']})")
            elif 'output_shape' in detail:
                print(f"    Output Shape: {detail['output_shape']}")
            print(f"    Latency: {detail['latency']:.6f}")
            print(f"    Cumulative: {detail['cumulative_latency']:.6f}")


if __name__ == "__main__":
    test_llama_layer_latency()

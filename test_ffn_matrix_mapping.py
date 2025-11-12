"""Test matrix mapping pass with FFNLayer.

This test:
1. Loads Meta-Llama-3-8B config
2. Initializes FFNLayer
3. Traces and converts to computation graph
4. Applies matrix fusion pass
5. Applies matrix mapping pass with H2LLM strategy
6. Outputs mapping results for each matrix operation
"""

import torch
from pathlib import Path

from pimapper.model.base import load_model_config, FFNLayer
from pimapper.modelmapper.converter import trace_module, fx_to_computation_graph
from pimapper.modelmapper.passes.normalize_ops import NormalizeOpsPass
from pimapper.modelmapper.passes.simplify import SimplifyGraphPass
from pimapper.modelmapper.passes.matrix_fusion import MatrixFusionPass
from pimapper.codegen.passes.matrix_mapping import MatrixMappingPass
from pimapper.core.hwspec import PIMChannelSpec, AcceleratorSpec


def main():
    print("=" * 80)
    print("FFN Layer Matrix Mapping Test")
    print("=" * 80)

    # Step 1: Load model configuration
    print("\n[Step 1] Loading model configuration...")
    card_path = Path("pimapper/model/model_cards/Meta-Llama-3-8B.json")
    config = load_model_config(card_path)
    print(f"  Model: Meta-Llama-3-8B")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Intermediate size: {config.intermediate_size}")

    # Step 2: Initialize FFNLayer
    print("\n[Step 2] Initializing FFNLayer...")
    ffn_layer = FFNLayer(config)
    ffn_layer.eval()
    print(f"  FFN Layer created with:")
    print(f"    - w1: {config.hidden_size} x {config.intermediate_size}")
    print(f"    - w2: {config.intermediate_size} x {config.hidden_size}")
    print(f"    - w3: {config.hidden_size} x {config.intermediate_size}")

    # Step 3: Trace and convert to computation graph
    print("\n[Step 3] Tracing FFNLayer to computation graph...")
    batch_size = 20
    seq_len = 1
    hidden_size = config.hidden_size
    # Use float16 to match FFNLayer dtype
    sample_input = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)

    # Trace the module
    fx_graph = trace_module(ffn_layer, sample_inputs=(sample_input,))
    print(f"  Traced {len(fx_graph.nodes)} FX nodes")

    # Convert to computation graph
    comp_graph = fx_to_computation_graph(fx_graph, ffn_layer)
    print(f"  Created computation graph with {len(list(comp_graph.nodes()))} nodes")

    # Step 4: Apply normalization and simplification passes
    print("\n[Step 4] Applying normalization and simplification passes...")
    normalize_pass = NormalizeOpsPass()
    normalize_pass.run(comp_graph)
    print(f"  Normalization complete")

    simplify_pass = SimplifyGraphPass()
    simplify_pass.run(comp_graph)
    print(f"  Simplification complete")
    print(f"  Graph now has {len(list(comp_graph.nodes()))} nodes")

    # Count matrix operations before fusion
    matmul_count_before = 0
    for node_name in comp_graph.nodes():
        op = comp_graph.node_record(node_name)
        if getattr(op, 'op_type', None) == 'matmul':
            matmul_count_before += 1
    print(f"  Matrix operations before fusion: {matmul_count_before}")

    # Step 5: Apply matrix fusion pass
    print("\n[Step 5] Applying matrix fusion pass...")
    fusion_pass = MatrixFusionPass(min_fusion_size=2, block_size=64)
    fusion_modified = fusion_pass.run(comp_graph)

    if fusion_modified:
        print(f"  Matrix fusion applied successfully")
        print(f"  Fusion groups created: {fusion_pass._metadata.get('fusion_groups', 0)}")
        print(f"  Total fused nodes: {fusion_pass._metadata.get('total_fused_nodes', 0)}")
    else:
        print(f"  No fusion opportunities found")

    # Count operations after fusion
    matmul_count_after = 0
    fusion_count = 0
    for node_name in comp_graph.nodes():
        op = comp_graph.node_record(node_name)
        op_type = getattr(op, 'op_type', None)
        if op_type == 'matmul':
            matmul_count_after += 1
        elif op_type == 'fusion_matrix':
            fusion_count += 1
    print(f"  Matrix operations after fusion: {matmul_count_after}")
    print(f"  Fusion matrix operations: {fusion_count}")

    # Step 6: Configure hardware
    print("\n[Step 6] Configuring hardware specification...")
    # Hardware: 5 channels, 4 TOPS compute, 0.4 TB/s memory bandwidth, 12.8 GB/s I/O bandwidth
    channel_spec = PIMChannelSpec(
        compute_power=4.0,  # 4 TOPS
        input_bandwidth=12.8,  # 12.8 GB/s
        output_bandwidth=12.8,  # 12.8 GB/s
        memory_bandwidth=400.0  # 0.4 TB/s = 400 GB/s
    )
    accelerator_spec = AcceleratorSpec(
        channel_count=8,
        channel_spec=channel_spec
    )
    print(f"  Accelerator configuration:")
    print(f"    - Channels: {accelerator_spec.channel_count}")
    print(f"    - Compute power per channel: {channel_spec.compute_power} TOPS")
    print(f"    - Memory bandwidth per channel: {channel_spec.memory_bandwidth} GB/s")
    print(f"    - Input bandwidth per channel: {channel_spec.input_bandwidth} GB/s")
    print(f"    - Output bandwidth per channel: {channel_spec.output_bandwidth} GB/s")

    # Step 7: Apply matrix mapping pass with recursive_grid_search strategy
    print("\n[Step 7] Applying matrix mapping pass (recursive_grid_search strategy)...")
    # Configure strategy parameters: search range [1,2,3,4,8]
    strategy_kwargs = {
        'num_split_row_candidates': [1, 2, 3, 4, 8],
        'num_split_col_candidates': [1, 2, 3, 4, 8],
        'max_iterations': 2
    }
    # mapping_pass = MatrixMappingPass(
    #     accelerator_spec=accelerator_spec,
    #     strategy="recursive_grid_search",
    #     strategy_kwargs=strategy_kwargs
    # )

    mapping_pass = MatrixMappingPass(
        accelerator_spec = accelerator_spec,
        strategy = "h2llm"
    )

    try:
        mapping_modified = mapping_pass.run(comp_graph)

        if mapping_modified:
            print(f"  Matrix mapping completed successfully")
            print(f"  Matrices mapped: {mapping_pass._metadata.get('matrices_mapped', 0)}")
            print(f"  Total latency: {mapping_pass._metadata.get('total_latency', 0.0):.6f} cycles")
        else:
            print(f"  No matrices to map")

    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return

    # Step 8: Display detailed mapping results
    print("\n[Step 8] Detailed Mapping Results")
    print("=" * 80)

    mapping_details = mapping_pass._metadata.get('mapping_details', [])

    if not mapping_details:
        print("  No mapping details available")
        return

    for idx, detail in enumerate(mapping_details, 1):
        node_name = detail['node_name']
        shape = detail['shape']
        latency = detail['latency']
        utilization = detail['utilization']

        print(f"\n  Matrix Operation #{idx}: {node_name}")
        print(f"    Shape: {shape[0]} x {shape[1]} (batch={shape[2]})")
        print(f"    Latency: {latency:.6f} cycles")
        print(f"    Compute Utilization: {utilization:.2%}")

        # Get the actual mapping result from the op
        op = comp_graph.node_record(node_name)
        if hasattr(op, 'kwargs') and 'mapping_result' in op.kwargs:
            mapping_result = op.kwargs['mapping_result']
            mapping = mapping_result.mapping

            print(f"    Mapping details:")
            print(f"      - Total tiles: {len(mapping.tiles)}")

            # Get channel tile counts from placement
            channel_tile_counts = {}
            for channel_id, tiles in mapping.placement.items():
                if tiles:
                    channel_tile_counts[channel_id] = len(tiles)

            print(f"      - Channels used: {len(channel_tile_counts)}")
            print(f"      - Tile distribution:")
            for channel_id in sorted(channel_tile_counts.keys()):
                count = channel_tile_counts[channel_id]
                tiles = mapping.placement[channel_id]
                # Show first few tiles as examples
                tile_examples = [f"{t.num_rows}×{t.num_cols}" for t in tiles[:3]]
                if len(tiles) > 3:
                    tile_examples.append(f"... +{len(tiles)-3} more")
                print(f"        {channel_id}: {count} tiles ({', '.join(tile_examples)})")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

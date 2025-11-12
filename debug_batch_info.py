"""Debug script to examine batch size calculation in matrix operations."""

import torch
from pathlib import Path

from pimapper.model.base import load_model_config, FFNLayer
from pimapper.modelmapper.converter import trace_module, fx_to_computation_graph
from pimapper.modelmapper.passes.normalize_ops import NormalizeOpsPass
from pimapper.modelmapper.passes.simplify import SimplifyGraphPass
from pimapper.modelmapper.passes.matrix_fusion import MatrixFusionPass


def main():
    print("=" * 80)
    print("Debugging Batch Size Calculation")
    print("=" * 80)

    # Load model and create FFN layer
    card_path = Path("pimapper/model/model_cards/Meta-Llama-3-8B.json")
    config = load_model_config(card_path)
    ffn_layer = FFNLayer(config)
    ffn_layer.eval()

    # Create sample input
    batch_size = 20
    seq_len = 4
    hidden_size = config.hidden_size
    sample_input = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)

    print(f"\nInput shape: {sample_input.shape}")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  hidden_size: {hidden_size}")

    # Trace and convert
    fx_graph = trace_module(ffn_layer, sample_inputs=(sample_input,))
    comp_graph = fx_to_computation_graph(fx_graph, ffn_layer)

    # Apply passes
    normalize_pass = NormalizeOpsPass()
    normalize_pass.run(comp_graph)

    simplify_pass = SimplifyGraphPass()
    simplify_pass.run(comp_graph)

    print("\n" + "=" * 80)
    print("BEFORE FUSION - Matrix Operations:")
    print("=" * 80)

    for node_name in comp_graph.nodes(sort=True):
        op = comp_graph.node_record(node_name)
        op_type = getattr(op, 'op_type', None)

        if op_type == 'matmul':
            print(f"\nNode: {node_name}")
            print(f"  op_type: {op_type}")

            # Check results
            if hasattr(op, 'results') and op.results:
                result = op.results[0]
                print(f"  results[0].shape: {result.shape}")
                print(f"  results[0].dtype: {result.dtype}")

            # Check metadata
            if hasattr(op, 'metadata') and op.metadata:
                custom = op.metadata.get('custom', {})
                weight_shape = custom.get('weight_shape')
                print(f"  metadata.custom.weight_shape: {weight_shape}")

            # Check kwargs
            if hasattr(op, 'kwargs'):
                transpose_b = op.kwargs.get('transpose_b', False)
                print(f"  kwargs.transpose_b: {transpose_b}")

    # Apply fusion
    print("\n" + "=" * 80)
    print("APPLYING FUSION...")
    print("=" * 80)

    fusion_pass = MatrixFusionPass(min_fusion_size=2, block_size=64)
    fusion_pass.run(comp_graph)

    print("\n" + "=" * 80)
    print("AFTER FUSION - Matrix Operations:")
    print("=" * 80)

    for node_name in comp_graph.nodes(sort=True):
        op = comp_graph.node_record(node_name)
        op_type = getattr(op, 'op_type', None)

        if op_type in ('matmul', 'fusion_matrix'):
            print(f"\nNode: {node_name}")
            print(f"  op_type: {op_type}")

            # Check results
            if hasattr(op, 'results') and op.results:
                result = op.results[0]
                print(f"  results[0].shape: {result.shape}")
                print(f"  results[0].dtype: {result.dtype}")

            # For fusion_matrix, check additional attributes
            if op_type == 'fusion_matrix':
                print(f"  fused_weight_shape: {op.fused_weight_shape}")
                print(f"  merge_tree.output_shape: {op.merge_tree.output_shape}")
                print(f"  merge_tree.weight_shape: {op.merge_tree.weight_shape}")

                # Check children
                if hasattr(op.merge_tree, 'children'):
                    print(f"  merge_tree.children count: {len(op.merge_tree.children)}")
                    for i, child in enumerate(op.merge_tree.children):
                        print(f"    child[{i}].output_shape: {child.output_shape}")
                        print(f"    child[{i}].weight_shape: {child.weight_shape}")

            # Check metadata
            if hasattr(op, 'metadata') and op.metadata:
                custom = op.metadata.get('custom', {})
                weight_shape = custom.get('weight_shape')
                print(f"  metadata.custom.weight_shape: {weight_shape}")

            # Check kwargs
            if hasattr(op, 'kwargs'):
                transpose_b = op.kwargs.get('transpose_b', False)
                print(f"  kwargs.transpose_b: {transpose_b}")

    print("\n" + "=" * 80)
    print("BATCH SIZE EXTRACTION LOGIC:")
    print("=" * 80)

    for node_name in comp_graph.nodes(sort=True):
        op = comp_graph.node_record(node_name)
        op_type = getattr(op, 'op_type', None)

        if op_type in ('matmul', 'fusion_matrix'):
            print(f"\nNode: {node_name} ({op_type})")

            # Simulate the extraction logic from MatrixMappingPass
            if op_type == 'fusion_matrix' and hasattr(op, 'fused_weight_shape'):
                rows, cols = op.fused_weight_shape
                print(f"  Using fused_weight_shape: {rows} x {cols}")

                # Get batch size from output shape
                if hasattr(op, 'results') and op.results:
                    result = op.results[0]
                    shape = getattr(result, 'shape', None)
                    print(f"  results[0].shape: {shape}")

                    if shape and len(shape) > 2:
                        batch_dims = shape[:-2]
                        import math
                        batch_size_calc = math.prod(batch_dims)
                        print(f"  batch_dims: {batch_dims}")
                        print(f"  CALCULATED batch_size: {batch_size_calc}")
                    elif shape and len(shape) == 2:
                        print(f"  2D shape, batch_size: {shape[0]}")
                    else:
                        print(f"  Default batch_size: 1")

            elif op_type == 'matmul':
                # Check metadata first
                if hasattr(op, 'metadata') and op.metadata:
                    custom = op.metadata.get('custom', {})
                    weight_shape = custom.get('weight_shape')

                    if weight_shape is not None:
                        transpose_b = op.kwargs.get('transpose_b', False) if hasattr(op, 'kwargs') else False

                        if transpose_b:
                            rows, cols = weight_shape[1], weight_shape[0]
                        else:
                            rows, cols = weight_shape

                        print(f"  Using metadata weight_shape: {rows} x {cols}")

                        # Get batch size from output shape
                        if hasattr(op, 'results') and op.results:
                            result = op.results[0]
                            output_shape = getattr(result, 'shape', None)
                            print(f"  results[0].shape: {output_shape}")

                            if output_shape and len(output_shape) > 2:
                                batch_dims = output_shape[:-1]  # All dimensions except the last one
                                import math
                                batch_size_calc = math.prod(batch_dims)
                                print(f"  batch_dims (all except last): {batch_dims}")
                                print(f"  CALCULATED batch_size: {batch_size_calc}")
                            elif output_shape and len(output_shape) == 2:
                                batch_size_calc = output_shape[0]
                                print(f"  2D shape, batch_size: {batch_size_calc}")


if __name__ == "__main__":
    main()

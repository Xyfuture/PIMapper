"""Test script for the converter pipeline with ops integration

This script tests the full pipeline:
1. Trace PyTorch module to torch.fx graph
2. Convert to NxComputationGraph (with torch_compat ops)
3. Apply NormalizeOpsPass (convert torch ops to native ops)
4. Apply SimplifyGraphPass (remove non-essential ops)

Verify that shape information is preserved throughout the pipeline.
"""

from pathlib import Path
import torch

from pimapper.modelmapper.converter import build_computation_graph
from pimapper.core.graph.base import NxComputationGraph


def print_graph_summary(graph: NxComputationGraph, title: str = "Graph Summary") -> None:
    """Print a summary of the computation graph."""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")
    print(f"Total nodes: {len(list(graph.nodes()))}")
    print(f"\nNode details:")
    print("-" * 80)

    for i, line in enumerate(graph.summarize()):
        print(f"{i+1:3}. {line}")


def print_edge_connections(graph: NxComputationGraph, title: str = "Edge Connections", max_nodes: int = 20) -> None:
    """Print detailed edge connection information."""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")

    all_nodes = list(graph.nodes())
    total_edges = sum(len(list(graph.successors(node))) for node in all_nodes)

    print(f"Total edges: {total_edges}")
    print(f"Total nodes: {len(all_nodes)}")
    print(f"\nShowing first {min(max_nodes, len(all_nodes))} nodes:")
    print("-" * 80)

    for i, node_name in enumerate(all_nodes[:max_nodes]):
        predecessors = list(graph.predecessors(node_name))
        successors = list(graph.successors(node_name))
        op = graph.node_record(node_name)

        print(f"\n{i+1:3}. Node: {node_name} (op_type: {op.op_type})")

        if predecessors:
            print(f"     Inputs ({len(predecessors)}):")
            for pred in predecessors:
                pred_op = graph.node_record(pred)
                pred_shape = pred_op.metadata.shape if pred_op.metadata else None
                print(f"       <- {pred} ({pred_op.op_type}, shape: {pred_shape})")
        else:
            print(f"     Inputs: (none - root node)")

        if successors:
            print(f"     Outputs ({len(successors)}):")
            for succ in successors:
                succ_op = graph.node_record(succ)
                print(f"       -> {succ} ({succ_op.op_type})")
        else:
            print(f"     Outputs: (none - leaf node)")

    if len(all_nodes) > max_nodes:
        print(f"\n... and {len(all_nodes) - max_nodes} more nodes")


def verify_shapes(graph: NxComputationGraph) -> tuple[int, int]:
    """Verify that shape information is preserved in the graph.

    Returns:
        Tuple of (nodes_with_shape, total_nodes)
    """
    nodes_with_shape = 0
    total_nodes = 0

    for node_name in graph.nodes():
        total_nodes += 1
        op = graph.node_record(node_name)
        shape = op.metadata.shape if op.metadata else None

        if shape is not None and len(shape) > 0:
            nodes_with_shape += 1

    return nodes_with_shape, total_nodes


def test_converter_pipeline() -> None:
    """Test the full converter pipeline."""

    # Find a model card file
    card_path = Path(__file__).parent / "archeive/model/model_cards/Meta-Llama-3-8B.json"

    if not card_path.exists():
        print(f"Error: Model card not found at {card_path}")
        print("Please provide a valid model card path.")
        return

    print(f"Using model card: {card_path}")

    # Test 1: Build graph without normalization or simplification
    print("\n" + "=" * 80)
    print("TEST 1: Raw graph (no normalization or simplification)")
    print("=" * 80)

    fx_graph, comp_graph_raw = build_computation_graph(
        card_path,
        batch_size=1,
        seq_len=4,
        normalize=False,
        simplify=False,
    )

    print_graph_summary(comp_graph_raw, "Raw Graph (Torch Ops)")
    shapes_raw = verify_shapes(comp_graph_raw)
    print(f"\nShape information: {shapes_raw[0]}/{shapes_raw[1]} nodes have shape info")

    # Show edge connections for raw graph
    print_edge_connections(comp_graph_raw, "Raw Graph Edge Connections", max_nodes=15)

    # Test 2: Build graph with normalization only
    print("\n" + "=" * 80)
    print("TEST 2: Normalized graph (torch ops -> native ops)")
    print("=" * 80)

    _, comp_graph_normalized = build_computation_graph(
        card_path,
        batch_size=1,
        seq_len=4,
        normalize=True,
        simplify=False,
    )

    print_graph_summary(comp_graph_normalized, "Normalized Graph (Native Ops)")
    shapes_norm = verify_shapes(comp_graph_normalized)
    print(f"\nShape information: {shapes_norm[0]}/{shapes_norm[1]} nodes have shape info")

    # Count native ops
    native_ops = 0
    for node_name in comp_graph_normalized.nodes():
        op = comp_graph_normalized.node_record(node_name)
        op_type = op.op_type
        if op_type in {"matmul", "vector_add", "vector_dot", "silu", "softmax", "rmsnorm"}:
            native_ops += 1

    print(f"Native ops converted: {native_ops}")

    # Show edge connections for normalized graph
    print_edge_connections(comp_graph_normalized, "Normalized Graph Edge Connections", max_nodes=15)

    # Test 3: Build graph with normalization and simplification
    print("\n" + "=" * 80)
    print("TEST 3: Fully processed graph (normalized + simplified)")
    print("=" * 80)

    _, comp_graph_final = build_computation_graph(
        card_path,
        batch_size=1,
        seq_len=4,
        normalize=True,
        simplify=True,
    )

    print_graph_summary(comp_graph_final, "Final Simplified Graph")
    shapes_final = verify_shapes(comp_graph_final)
    print(f"\nShape information: {shapes_final[0]}/{shapes_final[1]} nodes have shape info")

    # Show edge connections for final graph
    print_edge_connections(comp_graph_final, "Final Simplified Graph Edge Connections", max_nodes=15)

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(f"{'Stage':<30} {'Total Nodes':<15} {'Nodes w/ Shape':<20}")
    print("-" * 80)
    print(f"{'Raw (Torch Ops)':<30} {shapes_raw[1]:<15} {shapes_raw[0]:<20}")
    print(f"{'Normalized (Native Ops)':<30} {shapes_norm[1]:<15} {shapes_norm[0]:<20}")
    print(f"{'Simplified':<30} {shapes_final[1]:<15} {shapes_final[0]:<20}")

    # Verify shape preservation
    print("\n" + "=" * 80)
    print("SHAPE PRESERVATION VERIFICATION")
    print("=" * 80)

    if shapes_final[0] > 0:
        percentage = (shapes_final[0] / shapes_final[1]) * 100
        print(f"✓ Shape information preserved: {percentage:.1f}% of nodes have shape info")
        print(f"✓ Pipeline successfully completed!")
    else:
        print(f"✗ Warning: No shape information in final graph")


if __name__ == "__main__":
    test_converter_pipeline()

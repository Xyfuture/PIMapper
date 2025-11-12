"""Test for MatrixMappingPass.

This test verifies that the MatrixMappingPass correctly:
1. Identifies matrix operations in a computation graph
2. Maps them to hardware channels using a strategy
3. Stores mapping results in op.kwargs
4. Tracks statistics in metadata
"""

from pimapper.codegen.passes import MatrixMappingPass
from pimapper.core.hwspec import PIMChannelSpec, AcceleratorSpec
from pimapper.core.graph.base import NxComputationGraph
from pimapper.core.graph.ops.native import MatMulOp
from pimapper.core.graph.ops.base import GraphTensor
from pimapper.matrixmapper.strategy import RecursiveGridSearchStrategy


def test_matrix_mapping_pass():
    """Test MatrixMappingPass with a simple computation graph."""

    # Create hardware specification
    channel_spec = PIMChannelSpec(
        compute_power=100.0,
        shared_bandwidth=50.0,
        memory_bandwidth=1.0
    )
    accel_spec = AcceleratorSpec(channel_count=5, channel_spec=channel_spec)

    # Create computation graph
    graph = NxComputationGraph()

    # Add placeholder nodes (inputs)
    from pimapper.core.graph.ops.torch_compat import TorchPlaceholderOp

    input1_op = TorchPlaceholderOp(name="input1")
    input1_op.results = [GraphTensor(shape=(1, 4, 4096), dtype="torch.float16")]
    graph.create_node("input1", input1_op)

    input2_op = TorchPlaceholderOp(name="input2")
    input2_op.results = [GraphTensor(shape=(1, 4096, 12288), dtype="torch.float16")]
    graph.create_node("input2", input2_op)

    # Add MatMulOp node (batched 3D matmul)
    matmul1_op = MatMulOp(transpose_a=False, transpose_b=False)
    matmul1_op.results = [GraphTensor(shape=(1, 4, 12288), dtype="torch.float16")]
    matmul1_op.args = ("input1", "input2")
    graph.create_node("matmul1", matmul1_op)

    # Add another MatMulOp node (2D matmul)
    input3_op = TorchPlaceholderOp(name="input3")
    input3_op.results = [GraphTensor(shape=(2048, 4096), dtype="torch.float16")]
    graph.create_node("input3", input3_op)

    input4_op = TorchPlaceholderOp(name="input4")
    input4_op.results = [GraphTensor(shape=(4096, 8192), dtype="torch.float16")]
    graph.create_node("input4", input4_op)

    matmul2_op = MatMulOp(transpose_a=False, transpose_b=False)
    matmul2_op.results = [GraphTensor(shape=(2048, 8192), dtype="torch.float16")]
    matmul2_op.args = ("input3", "input4")
    graph.create_node("matmul2", matmul2_op)

    print("Created computation graph with 2 MatMulOp nodes")
    print(f"  - matmul1: shape (1, 4, 12288) - batched 3D")
    print(f"  - matmul2: shape (2048, 8192) - 2D")
    print()

    # Create and run the pass
    print("Creating MatrixMappingPass with RecursiveGridSearchStrategy...")
    strategy_kwargs = {
        "num_split_row_candidates": [1, 2, 4],
        "num_split_col_candidates": [1, 2, 4],
        "max_iterations": 1
    }

    pass_instance = MatrixMappingPass(
        accelerator_spec=accel_spec,
        strategy_kwargs=strategy_kwargs
    )

    print("Running pass on computation graph...")
    modified = pass_instance.run(graph)

    print(f"Pass completed. Graph modified: {modified}")
    print()

    # Verify results
    assert modified, "Pass should have modified the graph"

    # Check metadata
    metadata = pass_instance.get_metadata()
    print("Pass Metadata:")
    print(f"  - Matrices mapped: {metadata['matrices_mapped']}")
    print(f"  - Total latency: {metadata['total_latency']:.2f} cycles")
    print(f"  - Failed mappings: {metadata['failed_mappings']}")
    print()

    assert metadata['matrices_mapped'] == 2, "Should have mapped 2 matrices"
    assert metadata['total_latency'] > 0, "Total latency should be positive"
    assert len(metadata['failed_mappings']) == 0, "No mappings should have failed"

    # Check that mapping results are stored in op.kwargs
    print("Checking mapping results stored in operations:")
    for node_name in ["matmul1", "matmul2"]:
        op = graph.node_record(node_name)
        assert 'mapping_result' in op.kwargs, f"{node_name} should have mapping_result in kwargs"

        mapping_result = op.kwargs['mapping_result']
        assert mapping_result is not None, f"{node_name} mapping_result should not be None"
        assert hasattr(mapping_result, 'mapping'), "MappingResult should have mapping attribute"
        assert hasattr(mapping_result, 'latency'), "MappingResult should have latency attribute"

        print(f"  - {node_name}:")
        print(f"      Latency: {mapping_result.latency:.2f} cycles")
        print(f"      Utilization: {mapping_result.get_compute_utilization():.2%}")
        print(f"      Mapping tiles: {len(mapping_result.mapping.tiles)}")

    print()
    print("Detailed mapping information:")
    for detail in metadata['mapping_details']:
        print(f"  - {detail['node_name']}:")
        print(f"      Shape: {detail['shape']} (rows, cols, batch)")
        print(f"      Latency: {detail['latency']:.2f} cycles")
        print(f"      Utilization: {detail['utilization']:.2%}")

    print()
    print("All tests passed!")


if __name__ == "__main__":
    test_matrix_mapping_pass()

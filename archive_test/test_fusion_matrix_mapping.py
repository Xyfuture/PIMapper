"""Test FusionMatrixOp with MatrixMappingPass

This test verifies that:
1. FusionMatrixOp correctly inherits from MatMulOp
2. MergeTreeNode properly tracks both output_shape and weight_shape
3. MatrixMappingPass can recognize and map FusionMatrixOp
"""

from pimapper.core.graph.base import NxComputationGraph
from pimapper.core.graph.ops.fusionmatrix import FusionMatrixOp, MergeTreeNode, FusionStrategy
from pimapper.core.graph.ops.native import MatMulOp
from pimapper.core.graph.ops.base import GraphTensor
from pimapper.core.hwspec import PIMChannelSpec, AcceleratorSpec
from pimapper.codegen.passes.matrix_mapping import MatrixMappingPass


def test_merge_tree_shapes():
    """Test that MergeTreeNode correctly tracks output_shape and weight_shape"""
    print("\n=== Test 1: MergeTreeNode Shape Tracking ===")

    # Create leaf nodes with different shapes
    # Simulating: input (batch=2, seq=4, hidden=512) @ weight1 (512, 1024) -> (2, 4, 1024)
    leaf1 = MergeTreeNode.create_leaf(
        op_name="matmul1",
        op=MatMulOp(),
        shape=(2, 4, 1024),  # output shape
        latest_start_time=1.0
    )

    # Simulating: input (batch=2, seq=4, hidden=512) @ weight2 (512, 2048) -> (2, 4, 2048)
    leaf2 = MergeTreeNode.create_leaf(
        op_name="matmul2",
        op=MatMulOp(),
        shape=(2, 4, 2048),  # output shape
        latest_start_time=2.0
    )

    print(f"Leaf1 - output_shape: {leaf1.output_shape}, weight_shape: {leaf1.weight_shape}")
    print(f"Leaf2 - output_shape: {leaf2.output_shape}, weight_shape: {leaf2.weight_shape}")

    assert leaf1.output_shape == (2, 4, 1024), "Leaf1 output_shape incorrect"
    assert leaf1.weight_shape == (4, 1024), "Leaf1 weight_shape incorrect"
    assert leaf2.output_shape == (2, 4, 2048), "Leaf2 output_shape incorrect"
    assert leaf2.weight_shape == (4, 2048), "Leaf2 weight_shape incorrect"

    # Create internal node (fusion)
    internal = MergeTreeNode.create_internal(
        children=[leaf1, leaf2],
        strategy=FusionStrategy.SEQUENTIAL,
        block_size=1
    )

    print(f"Internal - output_shape: {internal.output_shape}, weight_shape: {internal.weight_shape}")

    # Fused output should be (2, 4, 1024+2048) = (2, 4, 3072)
    assert internal.output_shape == (2, 4, 3072), "Internal output_shape incorrect"
    # Fused weight should be (4, 1024+2048) = (4, 3072)
    assert internal.weight_shape == (4, 3072), "Internal weight_shape incorrect"

    print("[PASS] MergeTreeNode shape tracking works correctly")


def test_fusion_matrix_op_inheritance():
    """Test that FusionMatrixOp correctly inherits from MatMulOp"""
    print("\n=== Test 2: FusionMatrixOp Inheritance ===")

    # Create a simple merge tree
    leaf1 = MergeTreeNode.create_leaf(
        op_name="matmul1",
        op=MatMulOp(),
        shape=(2, 4, 1024),
        latest_start_time=1.0
    )

    leaf2 = MergeTreeNode.create_leaf(
        op_name="matmul2",
        op=MatMulOp(),
        shape=(2, 4, 2048),
        latest_start_time=2.0
    )

    root = MergeTreeNode.create_internal(
        children=[leaf1, leaf2],
        strategy=FusionStrategy.SEQUENTIAL
    )

    # Create FusionMatrixOp
    fusion_op = FusionMatrixOp(
        merge_tree=root,
        shared_inputs=["input_tensor"]
    )

    print(f"FusionMatrixOp is instance of MatMulOp: {isinstance(fusion_op, MatMulOp)}")
    print(f"FusionMatrixOp.op_type: {fusion_op.op_type}")
    print(f"FusionMatrixOp.fused_weight_shape: {fusion_op.fused_weight_shape}")
    print(f"FusionMatrixOp.results[0].shape: {fusion_op.results[0].shape}")

    assert isinstance(fusion_op, MatMulOp), "FusionMatrixOp should inherit from MatMulOp"
    assert fusion_op.op_type == "fusion_matrix", "op_type should be 'fusion_matrix'"
    assert fusion_op.fused_weight_shape == (4, 3072), "fused_weight_shape incorrect"
    assert fusion_op.results[0].shape == (2, 4, 3072), "output shape incorrect"

    print("[PASS] FusionMatrixOp inheritance works correctly")


def test_matrix_mapping_pass_with_fusion():
    """Test that MatrixMappingPass can recognize and map FusionMatrixOp"""
    print("\n=== Test 3: MatrixMappingPass with FusionMatrixOp ===")

    # Create a computation graph
    graph = NxComputationGraph()

    # Add input node
    from pimapper.core.graph.ops.torch_compat import TorchPlaceholderOp
    input_op = TorchPlaceholderOp(name="input")
    input_op.results = [GraphTensor(shape=(2, 4, 512), dtype="torch.float16")]
    graph.create_node("input", input_op)

    # Create two matmul operations
    matmul1 = MatMulOp()
    matmul1.results = [GraphTensor(shape=(2, 4, 1024), dtype="torch.float16")]
    graph.create_node("matmul1", matmul1)

    matmul2 = MatMulOp()
    matmul2.results = [GraphTensor(shape=(2, 4, 2048), dtype="torch.float16")]
    graph.create_node("matmul2", matmul2)

    # Create FusionMatrixOp
    leaf1 = MergeTreeNode.create_leaf(
        op_name="matmul1",
        op=matmul1,
        shape=(2, 4, 1024),
        latest_start_time=1.0
    )

    leaf2 = MergeTreeNode.create_leaf(
        op_name="matmul2",
        op=matmul2,
        shape=(2, 4, 2048),
        latest_start_time=2.0
    )

    root = MergeTreeNode.create_internal(
        children=[leaf1, leaf2],
        strategy=FusionStrategy.SEQUENTIAL
    )

    fusion_op = FusionMatrixOp(
        merge_tree=root,
        shared_inputs=["input"]
    )
    fusion_op.results = [GraphTensor(shape=(2, 4, 3072), dtype="torch.float16")]
    graph.create_node("fusion_matmul", fusion_op)

    # Create hardware spec
    channel_spec = PIMChannelSpec(
        compute_power=100.0,
        shared_bandwidth=50.0,
        memory_bandwidth=1.0
    )
    accel_spec = AcceleratorSpec(
        channel_count=4,
        channel_spec=channel_spec
    )

    # Create and run MatrixMappingPass
    mapping_pass = MatrixMappingPass(
        accelerator_spec=accel_spec,
        strategy="trivial"
    )

    print(f"Running MatrixMappingPass on graph with {len(list(graph.nodes()))} nodes...")

    try:
        modified = mapping_pass.run(graph)
        print(f"Pass modified graph: {modified}")

        # Check that fusion_matmul was mapped
        fusion_node = graph.node_record("fusion_matmul")
        if hasattr(fusion_node, 'kwargs') and 'mapping_result' in fusion_node.kwargs:
            mapping_result = fusion_node.kwargs['mapping_result']
            print(f"[PASS] FusionMatrixOp was successfully mapped!")
            print(f"  - Matrix shape: {mapping_result.mapping.matrix.rows} x {mapping_result.mapping.matrix.cols}")
            print(f"  - Batch size: {mapping_result.mapping.matrix.batch_size}")
            print(f"  - Latency: {mapping_result.latency:.4f}")
            print(f"  - Utilization: {mapping_result.get_compute_utilization():.2%}")

            # Verify the mapped shape matches fused_weight_shape
            assert mapping_result.mapping.matrix.rows == 4, "Mapped rows incorrect"
            assert mapping_result.mapping.matrix.cols == 3072, "Mapped cols incorrect"
            assert mapping_result.mapping.matrix.batch_size == 2, "Mapped batch_size incorrect"
        else:
            print("[FAIL] FusionMatrixOp was not mapped")
            return False

        # Check metadata
        metadata = mapping_pass._metadata
        print(f"\nPass metadata:")
        print(f"  - Matrices mapped: {metadata['matrices_mapped']}")
        print(f"  - Total latency: {metadata['total_latency']:.4f}")

        print("\n[PASS] MatrixMappingPass successfully recognizes and maps FusionMatrixOp")
        return True

    except Exception as e:
        print(f"[FAIL] Error during mapping: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_interleaved_fusion():
    """Test FusionMatrixOp with INTERLEAVED strategy"""
    print("\n=== Test 4: Interleaved Fusion Strategy ===")

    # Create leaf nodes
    leaf1 = MergeTreeNode.create_leaf(
        op_name="matmul1",
        op=MatMulOp(),
        shape=(1, 8, 512),
        latest_start_time=1.0
    )

    leaf2 = MergeTreeNode.create_leaf(
        op_name="matmul2",
        op=MatMulOp(),
        shape=(1, 8, 512),
        latest_start_time=2.0
    )

    # Create interleaved fusion
    root = MergeTreeNode.create_internal(
        children=[leaf1, leaf2],
        strategy=FusionStrategy.INTERLEAVED,
        block_size=64
    )

    print(f"Interleaved fusion - output_shape: {root.output_shape}, weight_shape: {root.weight_shape}")

    assert root.output_shape == (1, 8, 1024), "Interleaved output_shape incorrect"
    assert root.weight_shape == (8, 1024), "Interleaved weight_shape incorrect"

    # Create FusionMatrixOp
    fusion_op = FusionMatrixOp(
        merge_tree=root,
        shared_inputs=["input"]
    )

    print(f"FusionMatrixOp.fused_weight_shape: {fusion_op.fused_weight_shape}")
    assert fusion_op.fused_weight_shape == (8, 1024), "Interleaved fused_weight_shape incorrect"

    print("[PASS] Interleaved fusion strategy works correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing FusionMatrixOp Refactoring")
    print("=" * 60)

    try:
        test_merge_tree_shapes()
        test_fusion_matrix_op_inheritance()
        test_matrix_mapping_pass_with_fusion()
        test_interleaved_fusion()

        print("\n" + "=" * 60)
        print("[PASS] All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()

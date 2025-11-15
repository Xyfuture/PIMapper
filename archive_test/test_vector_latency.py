"""Test vector latency calculation pass.

Original prompt:
添加一个 Pass, 为 Vector 运算提供延迟仿真. 这个 pass 要能识别到所有的vector 运算.
这个 Pass 需要在 init 的时候输入一个 HostSpec, 在 @pimapper/core/hwspec.py 中,
用来获得 vector 算力的信息.

然后为每一个 vector op 加上一个 latency 的 kwargs, 用来记录这个 vector 的延迟.
延迟的计算的先用一个简单的方式, 就是根据这个 op 的输入, 计算出需要多少运算量
(需要考虑 batch size 和 vector 的长度), 然后根据 HostSpec 中的 vector 算力,
直接计算出时间.
"""

from pimapper.core.graph.base import NxComputationGraph
from pimapper.core.graph.ops.native import VectorAddOp, VectorMulOp, SiLUOp, SoftmaxOp, RMSNormOp
from pimapper.core.graph.ops.base import GraphTensor
from pimapper.core.hwspec import HostSpec, PIMChannelSpec, AcceleratorSpec, Accelerator
from pimapper.codegen.passes.vector_latency import VectorLatencyPass
from pimapper.codegen.passes.latency_calculation import LatencyCalculationPass


def test_vector_latency_simple():
    """Test vector latency calculation for a simple graph."""

    # Create a simple computation graph with vector operations
    graph = NxComputationGraph()

    # Create vector operations with different shapes
    # VectorAdd: batch=4, vector_length=1024
    vector_add = VectorAddOp(alpha=1.0)
    vector_add.results = [GraphTensor(shape=(4, 1024), dtype="torch.float16")]
    graph.create_node("vector_add_1", vector_add)

    # VectorMul: batch=4, vector_length=1024
    vector_mul = VectorMulOp()
    vector_mul.results = [GraphTensor(shape=(4, 1024), dtype="torch.float16")]
    graph.create_node("vector_mul_1", vector_mul)

    # SiLU: batch=4, vector_length=2048
    silu = SiLUOp()
    silu.results = [GraphTensor(shape=(4, 2048), dtype="torch.float16")]
    graph.create_node("silu_1", silu)

    # Softmax: batch=8, vector_length=512
    softmax = SoftmaxOp(dim=-1)
    softmax.results = [GraphTensor(shape=(8, 512), dtype="torch.float16")]
    graph.create_node("softmax_1", softmax)

    # RMSNorm: batch=2, seq_len=16, hidden=1024 -> batch=32, vector_length=1024
    rmsnorm = RMSNormOp(eps=1e-6)
    rmsnorm.results = [GraphTensor(shape=(2, 16, 1024), dtype="torch.float16")]
    graph.create_node("rmsnorm_1", rmsnorm)

    # Create HostSpec with 100 GFLOPS
    host_spec = HostSpec(vector_compute_power=100.0)

    print(f"Host Spec: {host_spec}")
    print(f"Graph has {len(graph.nodes())} nodes")

    # Run VectorLatencyPass
    vector_pass = VectorLatencyPass(host_spec)
    modified = vector_pass.run(graph)

    print(f"\nVectorLatencyPass modified: {modified}")
    print(f"Vectors processed: {vector_pass.get_metadata()['vectors_processed']}")
    print(f"Total latency: {vector_pass.get_metadata()['total_latency']:.2e} seconds")

    # Verify latencies were calculated
    assert modified, "VectorLatencyPass should have modified the graph"
    assert vector_pass.get_metadata()['vectors_processed'] == 5, "Should process 5 vector ops"

    # Manually verify calculations
    # VectorAdd: batch=4, vec_len=1024, flops_coef=1.0
    # total_flops = 4 * 1024 * 1.0 = 4096
    # latency = 4096 / (100 * 1e9) = 4.096e-8
    vector_add_latency = graph.node_record("vector_add_1").kwargs['latency']
    expected_latency = (4 * 1024 * 1.0) / (100.0 * 1e9)
    print(f"\nVectorAdd latency: {vector_add_latency:.2e} seconds (expected: {expected_latency:.2e})")
    assert abs(vector_add_latency - expected_latency) < 1e-12, "VectorAdd latency mismatch"

    # SiLU: batch=4, vec_len=2048, flops_coef=4.0
    # total_flops = 4 * 2048 * 4.0 = 32768
    # latency = 32768 / (100 * 1e9) = 3.2768e-7
    silu_latency = graph.node_record("silu_1").kwargs['latency']
    expected_latency = (4 * 2048 * 4.0) / (100.0 * 1e9)
    print(f"SiLU latency: {silu_latency:.2e} seconds (expected: {expected_latency:.2e})")
    assert abs(silu_latency - expected_latency) < 1e-12, "SiLU latency mismatch"

    # RMSNorm: batch=2*16=32, vec_len=1024, flops_coef=5.0
    # total_flops = 32 * 1024 * 5.0 = 163840
    # latency = 163840 / (100 * 1e9) = 1.6384e-6
    rmsnorm_latency = graph.node_record("rmsnorm_1").kwargs['latency']
    expected_latency = (2 * 16 * 1024 * 5.0) / (100.0 * 1e9)
    print(f"RMSNorm latency: {rmsnorm_latency:.2e} seconds (expected: {expected_latency:.2e})")
    assert abs(rmsnorm_latency - expected_latency) < 1e-12, "RMSNorm latency mismatch"

    # Run LatencyCalculationPass to verify total latency
    latency_calc_pass = LatencyCalculationPass()
    latency_calc_pass.run(graph)

    print(f"\nLatencyCalculationPass results:")
    print(f"Total latency: {latency_calc_pass.get_metadata()['total_latency']:.2e} seconds")
    print(f"Vector ops count: {latency_calc_pass.get_metadata()['vector_ops_count']}")
    print(f"Matrix ops count: {latency_calc_pass.get_metadata()['matrix_ops_count']}")
    print(f"Total ops count: {latency_calc_pass.get_metadata()['total_ops_count']}")

    assert latency_calc_pass.get_metadata()['vector_ops_count'] == 5, "Should count 5 vector ops"
    assert latency_calc_pass.get_metadata()['matrix_ops_count'] == 0, "Should count 0 matrix ops"
    assert latency_calc_pass.get_metadata()['total_ops_count'] == 5, "Should count 5 total ops"

    # Verify total latency matches sum of individual latencies
    total_expected = sum([
        graph.node_record("vector_add_1").kwargs['latency'],
        graph.node_record("vector_mul_1").kwargs['latency'],
        graph.node_record("silu_1").kwargs['latency'],
        graph.node_record("softmax_1").kwargs['latency'],
        graph.node_record("rmsnorm_1").kwargs['latency'],
    ])
    assert abs(latency_calc_pass.get_metadata()['total_latency'] - total_expected) < 1e-12, \
        "Total latency should match sum of individual latencies"

    print("\n[PASS] All tests passed!")


def test_accelerator_with_host_spec():
    """Test AcceleratorSpec and Accelerator with HostSpec."""

    # Create HostSpec
    host_spec = HostSpec(vector_compute_power=100.0)

    # Create PIMChannelSpec
    channel_spec = PIMChannelSpec(
        compute_power=100.0,  # 100 TFLOPS
        shared_bandwidth=50.0,  # 50 GB/s
        memory_bandwidth=1.0  # 1 TB/s
    )

    # Create AcceleratorSpec with host_spec
    accel_spec = AcceleratorSpec(
        channel_count=4,
        channel_spec=channel_spec,
        host_spec=host_spec
    )

    print(f"AcceleratorSpec: {accel_spec}")
    assert accel_spec.host_spec is not None, "AcceleratorSpec should have host_spec"
    assert accel_spec.host_spec.vector_compute_power == 100.0, "Host spec should have correct compute power"

    # Create Accelerator from spec
    accelerator = Accelerator.create_from_spec(accel_spec)

    print(f"Accelerator: {accelerator}")
    print(f"Host: {accelerator.host}")

    assert accelerator.host is not None, "Accelerator should have host"
    assert accelerator.host.spec is not None, "Host should have spec"
    assert accelerator.host.spec.vector_compute_power == 100.0, "Host spec should have correct compute power"

    print("\n[PASS] AcceleratorSpec test passed!")


def test_mixed_operations():
    """Test with both vector and matrix operations (when MatrixMappingPass is run)."""

    # Create a computation graph
    graph = NxComputationGraph()

    # Add vector operations
    vector_add = VectorAddOp(alpha=1.0)
    vector_add.results = [GraphTensor(shape=(4, 1024), dtype="torch.float16")]
    graph.create_node("vector_add_1", vector_add)

    silu = SiLUOp()
    silu.results = [GraphTensor(shape=(4, 2048), dtype="torch.float16")]
    graph.create_node("silu_1", silu)

    # Create HostSpec
    host_spec = HostSpec(vector_compute_power=50.0)

    # Run VectorLatencyPass
    vector_pass = VectorLatencyPass(host_spec)
    vector_pass.run(graph)

    # Run LatencyCalculationPass
    latency_calc_pass = LatencyCalculationPass()
    latency_calc_pass.run(graph)

    print(f"Mixed operations test:")
    print(f"Vector ops: {latency_calc_pass.get_metadata()['vector_ops_count']}")
    print(f"Matrix ops: {latency_calc_pass.get_metadata()['matrix_ops_count']}")
    print(f"Total latency: {latency_calc_pass.get_metadata()['total_latency']:.2e} seconds")

    assert latency_calc_pass.get_metadata()['vector_ops_count'] == 2, "Should count 2 vector ops"

    # Print detailed latency information
    print("\nDetailed latency information:")
    for detail in latency_calc_pass.get_metadata()['latency_details']:
        print(f"  {detail['node_name']} ({detail['category']}): {detail['latency']:.2e}s")

    print("\n[PASS] Mixed operations test passed!")


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Vector Latency Pass")
    print("=" * 80)

    test_vector_latency_simple()
    print("\n" + "=" * 80 + "\n")

    test_accelerator_with_host_spec()
    print("\n" + "=" * 80 + "\n")

    test_mixed_operations()
    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)

"""测试完整的转换流程"""

from pathlib import Path
from pimapper.modelmapper.converter import build_computation_graph
from pimapper.core.graph.base import NxComputationGraph


def print_graph_summary(graph: NxComputationGraph, title: str = "Graph Summary") -> None:
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")
    print(f"总节点数: {len(list(graph.nodes()))}")
    print(f"\n节点详情:")
    print("-" * 80)

    for i, line in enumerate(graph.summarize()):
        print(f"{i+1:3}. {line}")


def verify_shapes(graph: NxComputationGraph) -> tuple[int, int]:
    """验证 shape 信息是否保留"""
    nodes_with_shape = 0
    total_nodes = 0

    for node_name in graph.nodes():
        total_nodes += 1
        op = graph.node_record(node_name)
        if op.results and op.results[0].shape is not None:
            nodes_with_shape += 1

    return nodes_with_shape, total_nodes


def test_pipeline():
    card_path = Path("pimapper/model/model_cards/Meta-Llama-3-8B.json")

    if not card_path.exists():
        print(f"模型配置文件不存在: {card_path}")
        return

    print(f"使用模型配置: {card_path}")

    # 测试 1: 原始图 (torch ops)
    print("\n" + "=" * 80)
    print("测试 1: 原始图 (torch ops)")
    print("=" * 80)

    _, graph_raw = build_computation_graph(
        card_path,
        batch_size=1,
        seq_len=4,
        normalize=False,
        simplify=False,
    )

    print_graph_summary(graph_raw, "原始图 (Torch Ops)")
    shapes_raw = verify_shapes(graph_raw)
    print(f"\nShape 信息: {shapes_raw[0]}/{shapes_raw[1]} 个节点有 shape")

    # 测试 2: 归一化图 (native ops)
    print("\n" + "=" * 80)
    print("测试 2: 归一化图 (torch ops -> native ops)")
    print("=" * 80)

    _, graph_norm = build_computation_graph(
        card_path,
        batch_size=1,
        seq_len=4,
        normalize=True,
        simplify=False,
    )

    print_graph_summary(graph_norm, "归一化图 (Native Ops)")
    shapes_norm = verify_shapes(graph_norm)
    print(f"\nShape 信息: {shapes_norm[0]}/{shapes_norm[1]} 个节点有 shape")

    # 统计 native ops
    native_ops = 0
    for node_name in graph_norm.nodes():
        op = graph_norm.node_record(node_name)
        if op.op_type in {"matmul", "vector_add", "vector_mul", "silu", "softmax", "rmsnorm"}:
            native_ops += 1

    print(f"转换的 native ops: {native_ops}")

    # 测试 3: 完整处理图 (normalized + simplified)
    print("\n" + "=" * 80)
    print("测试 3: 完整处理图 (normalized + simplified)")
    print("=" * 80)

    _, graph_final = build_computation_graph(
        card_path,
        batch_size=1,
        seq_len=4,
        normalize=True,
        simplify=True,
    )

    print_graph_summary(graph_final, "最终简化图")
    shapes_final = verify_shapes(graph_final)
    print(f"\nShape 信息: {shapes_final[0]}/{shapes_final[1]} 个节点有 shape")

    # 对比总结
    print("\n" + "=" * 80)
    print("对比总结")
    print("=" * 80)
    print(f"{'阶段':<30} {'总节点数':<15} {'有 Shape 的节点':<20}")
    print("-" * 80)
    print(f"{'原始 (Torch Ops)':<30} {shapes_raw[1]:<15} {shapes_raw[0]:<20}")
    print(f"{'归一化 (Native Ops)':<30} {shapes_norm[1]:<15} {shapes_norm[0]:<20}")
    print(f"{'简化':<30} {shapes_final[1]:<15} {shapes_final[0]:<20}")

    # 验证 shape 保留
    print("\n" + "=" * 80)
    print("Shape 保留验证")
    print("=" * 80)

    if shapes_final[0] > 0:
        percentage = (shapes_final[0] / shapes_final[1]) * 100
        print(f"✓ Shape 信息已保留: {percentage:.1f}% 的节点有 shape 信息")
        print(f"✓ 流程成功完成!")
    else:
        print(f"✗ 警告: 最终图中没有 shape 信息")


if __name__ == "__main__":
    test_pipeline()

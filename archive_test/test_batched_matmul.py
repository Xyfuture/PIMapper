"""测试 BatchedMatMulWithPast 模块和 BatchedMatMulOp 的集成

原始 Prompt:
实现基础的 BatchedMatMulWithPast

请你修改 @pimapper/model/base.py 中的 llamalayer. 现在其代码中 attention 部分是直接运算的, 现在要进行修改, 使用 nn.module 实现 attention中的运算

首先创建一个 class 叫做 inference config , 里面简单设置两个参数, 一个是 batch size, 另一个是 past sequence length. 默认值一个是 1 一个是 1024.

然后创建一个 BatchedMatMulWithPast 的 class, 继承自 nn.Module, 具体的要求如下

- 里面有一个 past matrix, 表示记录的以前的内容
-  forward, 函数需要两个入参, 一个是 input, 另一个是 coming_kv.  forward 运行中, coming_kv 不需要任何操作, input 则需要与 past_matrix 进行 batched Matmul 操作, 得出一个矩阵.
- 我要用这个 module 替换掉 原来的直接相乘, 因此 输入/输出的 dim 要与原来保持一直, 你要读取一下 ModelConfig中的信息 配置num_attention_heads 和 num_key_value_heads, 注意 GQA的情况,
- 你要根据 Inference Config和ModelConfig 来记录一下 past matrix 的维度, 注意, 对于 QK 运算和 KT 运算应该是有区别的, __init__ 的时候通过一个参数记录一下

参考 RotaryPositionEmbedding 在 @pimapper/modelmapper/converter.py 中 NxGraph 的待遇, 将其实现标记为一个 leaf module , 然后在 @pimapper/core/graph/ops/native.py 中为其创建一个 BatchedMatMulOp 的class, 用于记录 BatchedMatMul 的一些基础信息, 需要记录

- Batch的大小, 也就是有多少个 matmul
- 假设所有的MatMul 都是相同的, 记录这个MatMul的存储 (M, rows, cols)
- 记录一下 ModelConfig 和 InferenceConfig.

结束之后编写一个测试样例, 参考 @test_full_pipeline.py  确保这个新引入的Op 能够存活在 **build_computation_graph** 之后, 同时输出一下最终的计算图的信息
"""

from pathlib import Path
from pimapper.modelmapper.converter import build_computation_graph
from pimapper.core.graph.base import NxComputationGraph
from pimapper.model.base import InferenceConfig


def print_graph_summary(graph: NxComputationGraph, title: str = "Graph Summary") -> None:
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")
    print(f"总节点数: {len(list(graph.nodes()))}")
    print(f"\n节点详情:")
    print("-" * 80)

    for i, line in enumerate(graph.summarize()):
        print(f"{i+1:3}. {line}")


def count_op_types(graph: NxComputationGraph) -> dict[str, int]:
    """统计各种操作类型的数量"""
    op_counts = {}
    for node_name in graph.nodes():
        op = graph.node_record(node_name)
        op_type = op.op_type
        op_counts[op_type] = op_counts.get(op_type, 0) + 1
    return op_counts


def verify_batched_matmul_ops(graph: NxComputationGraph) -> tuple[int, list[str]]:
    """验证 BatchedMatMulOp 是否存在"""
    batched_matmul_count = 0
    batched_matmul_nodes = []

    for node_name in graph.nodes():
        op = graph.node_record(node_name)
        if op.op_type == "batched_matmul":
            batched_matmul_count += 1
            batched_matmul_nodes.append(node_name)

    return batched_matmul_count, batched_matmul_nodes


def print_batched_matmul_details(graph: NxComputationGraph, node_names: list[str]) -> None:
    """打印 BatchedMatMulOp 的详细信息"""
    print(f"\n{'=' * 80}")
    print("BatchedMatMulOp 详细信息")
    print(f"{'=' * 80}")

    for i, node_name in enumerate(node_names, 1):
        op = graph.node_record(node_name)
        print(f"\n{i}. Node: {node_name}")
        print(f"   Op Type: {op.op_type}")
        print(f"   Kwargs: {op.kwargs}")
        if op.results:
            print(f"   Output Shape: {op.results[0].shape}")
            print(f"   Output Dtype: {op.results[0].dtype}")


def test_batched_matmul_integration():
    """测试 BatchedMatMulWithPast 和 BatchedMatMulOp 的集成"""
    card_path = Path("pimapper/model/model_cards/Meta-Llama-3-8B.json")

    if not card_path.exists():
        print(f"模型配置文件不存在: {card_path}")
        return

    print(f"使用模型配置: {card_path}")

    inference_config = InferenceConfig(batch_size=1, past_seq_len=1024)

    # 测试 1: 原始图 (torch ops) - 不进行归一化
    print("\n" + "=" * 80)
    print("测试 1: 原始图 (torch ops, 未归一化)")
    print("=" * 80)

    _, graph_raw = build_computation_graph(
        card_path,
        inference_config=inference_config,
        normalize=False,
        simplify=False,
    )

    print_graph_summary(graph_raw, "原始图 (Torch Ops)")

    # 检查是否有 BatchedMatMulWithPast 的 call_module
    batched_matmul_modules = 0
    for node_name in graph_raw.nodes():
        op = graph_raw.node_record(node_name)
        if op.op_type == "call_module":
            if op.metadata and op.metadata.get("custom", {}).get("module_class") == "BatchedMatMulWithPast":
                batched_matmul_modules += 1

    print(f"\n找到 {batched_matmul_modules} 个 BatchedMatMulWithPast call_module 节点")

    # 测试 2: 归一化图 (native ops)
    print("\n" + "=" * 80)
    print("测试 2: 归一化图 (torch ops -> native ops)")
    print("=" * 80)

    _, graph_norm = build_computation_graph(
        card_path,
        inference_config=inference_config,
        normalize=True,
        simplify=False,
    )

    print_graph_summary(graph_norm, "归一化图 (Native Ops)")

    # 统计操作类型
    op_counts = count_op_types(graph_norm)
    print(f"\n操作类型统计:")
    print("-" * 80)
    for op_type, count in sorted(op_counts.items()):
        print(f"{op_type:<30} {count:>5}")

    # 验证 BatchedMatMulOp
    batched_count, batched_nodes = verify_batched_matmul_ops(graph_norm)
    print(f"\n找到 {batched_count} 个 BatchedMatMulOp 节点")

    if batched_count > 0:
        print_batched_matmul_details(graph_norm, batched_nodes)

    # 测试 3: 完整处理图 (normalized + simplified)
    print("\n" + "=" * 80)
    print("测试 3: 完整处理图 (normalized + simplified)")
    print("=" * 80)

    _, graph_final = build_computation_graph(
        card_path,
        inference_config=inference_config,
        normalize=True,
        simplify=True,
    )

    print_graph_summary(graph_final, "最终简化图")

    # 验证 BatchedMatMulOp 在简化后是否仍然存在
    batched_count_final, batched_nodes_final = verify_batched_matmul_ops(graph_final)
    print(f"\n简化后找到 {batched_count_final} 个 BatchedMatMulOp 节点")

    if batched_count_final > 0:
        print_batched_matmul_details(graph_final, batched_nodes_final)

    # 对比总结
    print("\n" + "=" * 80)
    print("对比总结")
    print("=" * 80)
    print(f"{'阶段':<30} {'总节点数':<15} {'BatchedMatMulOp 数量':<25}")
    print("-" * 80)
    print(f"{'原始 (Torch Ops)':<30} {len(list(graph_raw.nodes())):<15} {batched_matmul_modules:<25}")
    print(f"{'归一化 (Native Ops)':<30} {len(list(graph_norm.nodes())):<15} {batched_count:<25}")
    print(f"{'简化':<30} {len(list(graph_final.nodes())):<15} {batched_count_final:<25}")

    # 验证结果
    print("\n" + "=" * 80)
    print("验证结果")
    print("=" * 80)

    if batched_count_final > 0:
        print(f"[PASS] BatchedMatMulOp 成功存活在 build_computation_graph 之后!")
        print(f"[PASS] 在最终图中找到 {batched_count_final} 个 BatchedMatMulOp 节点")
        print(f"[PASS] 测试通过!")
    else:
        print(f"[FAIL] 警告: 最终图中没有找到 BatchedMatMulOp 节点")
        print(f"[FAIL] 可能的原因:")
        print(f"  - BatchedMatMulOp 在简化过程中被移除")
        print(f"  - 转换过程中出现问题")


if __name__ == "__main__":
    test_batched_matmul_integration()

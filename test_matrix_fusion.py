"""测试矩阵融合 Pass - 使用 LLaMA 模型"""

from pathlib import Path
from pimapper.modelmapper.converter import build_computation_graph
from pimapper.modelmapper.passes.matrix_fusion import MatrixFusionPass


def test_matrix_fusion_on_llama():
    """在 LLaMA 模型上测试矩阵融合"""
    card_path = Path("pimapper/model/model_cards/Meta-Llama-3-8B.json")

    if not card_path.exists():
        print(f"模型配置文件不存在: {card_path}")
        return

    print("=" * 80)
    print("测试矩阵融合 Pass - LLaMA 模型")
    print("=" * 80)

    # 构建归一化图（包含 native ops）
    _, graph = build_computation_graph(
        card_path,
        batch_size=1,
        seq_len=4,
        normalize=True,
        simplify=True,
    )

    print(f"\n融合前:")
    print(f"  节点数: {len(list(graph.nodes()))}")

    # 统计矩阵运算节点
    matmul_nodes = [n for n in graph.nodes() if graph.node_record(n).op_type == "matmul"]
    print(f"  矩阵运算节点: {len(matmul_nodes)}")

    # 运行矩阵融合 Pass
    fusion_pass = MatrixFusionPass(min_fusion_size=2, block_size=64)
    modified = fusion_pass.run(graph)

    print(f"\n融合后:")
    print(f"  图是否被修改: {modified}")
    print(f"  节点数: {len(list(graph.nodes()))}")

    # 统计融合节点
    fusion_nodes = [n for n in graph.nodes() if graph.node_record(n).op_type == "fusion_matrix"]
    print(f"  融合节点数: {len(fusion_nodes)}")
    print(f"  Pass 元数据: {fusion_pass._metadata}")

    # 打印融合节点详情
    if fusion_nodes:
        print(f"\n融合节点详情:")
        print("-" * 80)
        for node_name in fusion_nodes:
            op = graph.node_record(node_name)
            print(f"\n  {node_name}:")
            print(f"    共享输入: {op.shared_inputs}")
            print(f"    融合的矩阵数: {len(op.get_all_original_ops())}")
            print(f"    融合策略: {op.merge_tree.strategy}")
            print(f"    原始 Op:")
            for orig_name, orig_op in op.get_all_original_ops():
                result_filter = op.get_result_filter(orig_name)
                print(f"      - {orig_name}: {orig_op.op_type}, filter={result_filter}")

    # 输出最终图结构
    print(f"\n最终图结构:")
    print("-" * 80)
    for i, line in enumerate(graph.summarize(), 1):
        print(f"{i:3}. {line}")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_matrix_fusion_on_llama()

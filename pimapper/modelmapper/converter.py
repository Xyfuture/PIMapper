from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn.modules import Module

from pimapper.core.graph.base import NxComputationGraph
from pimapper.model.base import RotaryPositionEmbedding, load_model_config, initialize_module
from pimapper.core.graph.ops.base import OpMetadata
from pimapper.core.graph.ops.torch_compat import create_torch_op_from_fx
from pimapper.modelmapper.passes.normalize_ops import NormalizeOpsPass
from pimapper.modelmapper.passes.simplify import SimplifyGraphPass


class NxGraphTracer(fx.Tracer):

    def is_leaf_module(self, m: Module, module_qualified_name: str) -> bool:
        if isinstance(m,RotaryPositionEmbedding):
            return True
        return super().is_leaf_module(m, module_qualified_name) 


def _create_sample_inputs(module: torch.nn.Module, batch_size: int = 1, seq_len: int = 4) -> tuple[torch.Tensor, ...]:
    """Create sample inputs for shape propagation."""
    hidden_size = module.config.hidden_size
    return (torch.randn(batch_size, seq_len, hidden_size),)


def trace_module(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[dict[str, Any]] = None,
    sample_inputs: Optional[tuple[torch.Tensor, ...]] = None,
    *,
    batch_size: int = 1,
    seq_len: int = 4,
) -> fx.Graph:
    """Symbolically trace a module and return the computation graph with shape information."""
    tracer = NxGraphTracer()
    graph = tracer.trace(root, concrete_args)

    # Create GraphModule to enable shape propagation
    gm = fx.GraphModule(root, graph)

    # Use ShapeProp to add shape information to the graph
    shape_prop = ShapeProp(gm)

    # If sample inputs are provided, use them; otherwise create dummy inputs
    if sample_inputs is None:
        # Try to create symbolic batch size or use default values
        sample_inputs = _create_sample_inputs(root, batch_size=batch_size, seq_len=seq_len)

    try:
        shape_prop.propagate(*sample_inputs)
    except Exception:
        pass

    return graph





def _tensor_meta_to_dict(tensor_meta: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "shape": tuple(getattr(tensor_meta, "shape", ())),
        "dtype": str(getattr(tensor_meta, "dtype", "")),
        "requires_grad": bool(getattr(tensor_meta, "requires_grad", False)),
    }
    device = getattr(tensor_meta, "device", None)
    if device is not None:
        meta["device"] = str(device)
    stride = getattr(tensor_meta, "stride", None)
    if stride is not None:
        meta["stride"] = tuple(stride)
    numel = getattr(tensor_meta, "numel", None)
    if callable(numel):
        try:
            meta["numel"] = tensor_meta.numel()
        except Exception:  # pragma: no cover - defensive
            pass
    return meta


def _sanitize_meta(node: fx.Node) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for key, value in node.meta.items():
        if key == "tensor_meta" and value is not None:
            meta[key] = _tensor_meta_to_dict(value)
        else:
            meta[key] = value
    if node.type is not None:
        meta.setdefault("type", str(node.type))
    return meta


def fx_to_computation_graph(graph: fx.Graph, module: Optional[torch.nn.Module] = None) -> NxComputationGraph:
    """Convert a torch.fx.Graph into a ComputationGraph using torch_compat ops.

    This function converts torch.fx graph nodes to torch_compat op representations,
    which can then be normalized to native ops via NormalizeOpsPass.

    Edges are automatically created by NxComputationGraph based on node references in args/kwargs.
    """
    comp_graph = NxComputationGraph()

    # Get module mapping if available
    modules = dict(module.named_modules()) if module is not None else {}

    # Helper function to convert fx.Node references to string names
    def convert_node_refs(value: Any) -> Any:
        """Recursively convert fx.Node objects to their string names."""
        if isinstance(value, fx.Node):
            return value.name
        elif isinstance(value, tuple):
            return tuple(convert_node_refs(item) for item in value)
        elif isinstance(value, list):
            return [convert_node_refs(item) for item in value]
        elif isinstance(value, dict):
            return {k: convert_node_refs(v) for k, v in value.items()}
        return value

    for node in graph.nodes:
        # Get shape information from ShapeProp (stored as node.shape, node.dtype)
        shape = getattr(node, 'shape', None)
        dtype = getattr(node, 'dtype', None)

        # Fallback to tensor_meta if shape/dtype not directly available
        if shape is None or dtype is None:
            tensor_meta = node.meta.get("tensor_meta")
            if tensor_meta is not None:
                shape = getattr(tensor_meta, 'shape', shape)
                dtype = getattr(tensor_meta, 'dtype', dtype)

        # Convert to tuple if needed
        if shape is not None and not isinstance(shape, tuple):
            shape = tuple(shape)
        if dtype is not None:
            dtype = str(dtype)

        # Enhanced metadata for module calls
        meta = _sanitize_meta(node)

        if node.op == 'call_module' and node.target in modules:
            target_module = modules[node.target]
            # Add module class name
            meta['module_class'] = target_module.__class__.__name__

            # Add weight shape information for linear modules
            if isinstance(target_module, torch.nn.Linear):
                if hasattr(target_module, 'weight') and target_module.weight is not None:
                    meta['weight_shape'] = tuple(target_module.weight.shape)
                if hasattr(target_module, 'bias') and target_module.bias is not None:
                    meta['bias_shape'] = tuple(target_module.bias.shape)
                # Also store in/out features for convenience
                meta['in_features'] = target_module.in_features
                meta['out_features'] = target_module.out_features

        # Create OpMetadata with shape and dtype information
        op_metadata = OpMetadata(
            shape=shape,
            dtype=dtype,
            custom=meta
        )

        # Convert fx.Node references to string names in args and kwargs
        converted_args = convert_node_refs(node.args)
        converted_kwargs = convert_node_refs(node.kwargs)

        # Create torch_compat Op from fx node with converted args/kwargs
        op = create_torch_op_from_fx(
            node.op,
            node.target,
            args=converted_args,
            kwargs=converted_kwargs,
            metadata=op_metadata
        )

        # Create node in computation graph with Op object
        # Edges will be automatically created from op.get_input_refs()
        comp_graph.create_node(name=node.name, op=op)

    return comp_graph


def build_computation_graph(
    card_path: str | Path,
    *,
    batch_size: int = 1,
    seq_len: int = 4,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    normalize: bool = True,
    simplify: bool = True,
) -> tuple[fx.Graph, NxComputationGraph]:
    """Full pipeline: load config, init module, trace, convert, normalize, and simplify.

    Args:
        card_path: Path to model configuration JSON file
        batch_size: Batch size for sample inputs
        seq_len: Sequence length for sample inputs
        dtype: Data type for sample inputs
        device: Device for sample inputs
        normalize: Whether to apply NormalizeOpsPass (convert torch ops to native ops)
        simplify: Whether to apply SimplifyGraphPass (remove non-essential ops)

    Returns:
        Tuple of (torch.fx.Graph, NxComputationGraph)
        The NxComputationGraph will have normalized and/or simplified ops based on parameters.
    """
    config = load_model_config(card_path)
    module = initialize_module(config, dtype=dtype, device=device)

    hidden_size = config.hidden_size
    sample_input = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)

    # Step 1: Trace the module to get torch.fx graph
    graph = trace_module(module, sample_inputs=(sample_input,))

    # Step 2: Convert to NxComputationGraph (with torch_compat ops)
    comp_graph = fx_to_computation_graph(graph, module)

    # Step 3: Apply NormalizeOpsPass (torch_compat -> native ops)
    if normalize:
        normalize_pass = NormalizeOpsPass()
        normalize_pass.run(comp_graph)

    # Step 4: Apply SimplifyGraphPass (remove non-essential ops)
    if simplify:
        simplify_pass = SimplifyGraphPass()
        simplify_pass.run(comp_graph)

    return graph, comp_graph


def simplify_computation_graph(graph: NxComputationGraph) -> NxComputationGraph:
    """简化计算图，保留核心操作并过滤掉不必要的操作。

    保留的操作类型：
    1. 神经网络模块 (call_module):
       - 线性层、注意力层、归一化层等核心计算模块
       - 过滤掉 view-like 模块（如 reshape、flatten 等）

    2. 核心张量操作 (call_function):
       - 激活函数: silu, sigmoid, tanh, relu, gelu
       - 基础线性代数: add, sub, mul, div, matmul, bmm
       - 向量操作: sum, mean, max, min
       - RoPE相关: sin, cos (旋转位置编码)
       - 基础形状操作: transpose, permute, contiguous
       - 索引操作: index_select, gather, scatter

    3. 方法调用 (call_method):
       - 张量方法: add, sub, mul, div, matmul
       - 聚合方法: sum, mean, max, min
       - 形状方法: transpose, permute, contiguous
       - 索引方法: index_select, gather, scatter
       - 查询方法: size, shape

    4. 输入占位符 (placeholder):
       - 模型输入节点

    过滤掉的操作：
    - 视图操作: view, reshape, flatten, squeeze, unsqueeze
    - 扩展操作: expand, repeat, tile
    这些操作不改变实际数据，只是改变张量的视图或形状

    注意：此函数现在在原图上进行修改，使用 replace_input 和 replace_uses 来删除节点
    """
    # 定义需要保留的核心操作类型
    ESSENTIAL_OPERATIONS = {
        # 神经网络模块 - 模型的核心计算组件
        'call_module',

        # 核心张量操作 - 数学运算和函数调用
        'call_function',
        'call_method',
        'output',

        # 占位符节点 - 模型输入
        'placeholder'
    }

    # 定义需要保留的具体函数 (基础线性向量运算)
    ESSENTIAL_FUNCTIONS = {
        # 激活函数 - 非线性变换核心
        'silu', 'sigmoid', 'tanh', 'relu', 'gelu',

        # 基础线性代数运算
        'add', 'sub', 'mul', 'div', 'matmul', 'bmm',

        # 向量聚合操作
        'sum', 'mean', 'max', 'min',

        # RoPE (旋转位置编码) 相关三角函数
        'sin', 'cos',

        # 基础形状操作 (保留改变数据排列的操作，过滤视图操作)
        # 'transpose', 'permute', 'contiguous',

        # 索引和切片操作
        # 'index_select', 'gather', 'scatter'
    }

    # 定义需要保留的特定方法
    ESSENTIAL_METHODS = {
        # 张量运算方法
        'add', 'sub', 'mul', 'div', 'matmul',
        # 聚合方法
        'sum', 'mean', 'max', 'min',
        # 形状变换方法
        # 'transpose', 'permute', 'contiguous',
        # 索引操作方法
        # 'index_select', 'gather', 'scatter',
        # 'size', 'shape'  # 形状查询方法
    }

    # 定义需要过滤掉的操作 (不改变实际数据的视图操作)
    FILTERED_OPERATIONS = {
        'view', 'reshape', 'flatten', 'squeeze', 'unsqueeze',  # 视图操作
        'expand', 'repeat', 'tile',  # 扩展和重复操作
        'transpose', 'permute', 'contiguous',  # 形状变换操作
        'getattr', 'getitem', 'setitem',  # 属性访问和索引操作
    }

    def should_keep_node(node_name: str) -> bool:
        """判断节点是否应该保留"""
        op = graph.node_record(node_name)
        op_type = op.op_type

        if op_type not in ESSENTIAL_OPERATIONS:
            return False

        if op_type == 'call_module':
            # 始终保留模块，但过滤掉视图类模块
            target = getattr(op, 'module_name', None) or getattr(op, 'target', None)
            if isinstance(target, str) and any(filtered in target.lower() for filtered in FILTERED_OPERATIONS):
                return False
            return True

        elif op_type == 'call_function':
            target = getattr(op, 'target', None)
            target_str = str(target).lower() if target else ''

            # 检查是否为保留的特定函数
            if any(essential in target_str for essential in ESSENTIAL_FUNCTIONS):
                return True

            # 检查是否包含过滤操作
            if any(filtered in target_str for filtered in FILTERED_OPERATIONS):
                return False

            # 其他函数默认过滤
            return False

        elif op_type == 'call_method':
            target = getattr(op, 'method_name', None) or getattr(op, 'target', None)
            target_str = str(target).lower() if target else ''

            # 检查是否为保留的特定方法
            if any(essential in target_str for essential in ESSENTIAL_METHODS):
                return True

            # 检查是否包含过滤操作
            if any(filtered in target_str for filtered in FILTERED_OPERATIONS):
                return False

            # 其他方法默认过滤
            return False

        elif op_type == 'placeholder':
            return True

        return False

    # 第一遍：标记需要删除的节点
    nodes_to_remove = []
    for node_name in graph.nodes(sort=False):
        if not should_keep_node(node_name):
            nodes_to_remove.append(node_name)

    # 简化计算图：准备删除 {len(nodes_to_remove)} 个节点

    # 第二遍：安全删除节点，使用 replace_input 和 replace_uses
    removed_count = 0
    for node_name in nodes_to_remove:
        if node_name not in graph.graph:  # 节点可能已经被删除
            continue

        op = graph.node_record(node_name)
        op_type = op.op_type
        target = getattr(op, 'target', None)

        # 获取节点的前驱和后继
        predecessors = list(graph.predecessors(node_name))
        successors = list(graph.successors(node_name))

        # 正在删除节点 '{node_name}'

        if op_type == 'call_function' and any(filtered in str(target).lower() for filtered in {'getattr', 'getitem', 'setitem'}):
            graph.remove_node(node_name, safe=False)
            removed_count += 1
            continue

        if len(predecessors) == 0:
            graph.remove_node(node_name, safe=False)
            removed_count += 1
        elif len(predecessors) == 1:
            graph.replace_uses(node_name, predecessors[0])
            graph.remove_node(node_name, safe=False)
            removed_count += 1
        else:
            # >=1
            assert False, f"节点 '{node_name}' 前驱数量大于 1，请检查逻辑"

    # 简化完成：共删除了 {removed_count} 个节点
    return graph


def summarize_graph(graph: NxComputationGraph) -> list[str]:
    lines: list[str] = []
    for name in graph.nodes():
        op = graph.node_record(name)
        shape = op.metadata.shape if op.metadata else None
        dtype = op.metadata.dtype if op.metadata else None

        # 获取后继节点
        successors = list(graph.successors(name))
        if successors:
            succ_str = ", ".join(successors)
            lines.append(f"{name}: {op.op_type} -> {shape} {dtype} -> [{succ_str}]")
        else:
            lines.append(f"{name}: {op.op_type} -> {shape} {dtype}")
    return lines




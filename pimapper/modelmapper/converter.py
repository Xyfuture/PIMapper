from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn.modules import Module

from pimapper.core.graph.base import NxComputationGraph
from pimapper.model.base import RotaryPositionEmbedding, BatchedMatMulWithPast, load_model_config, initialize_module, InferenceConfig
from pimapper.core.graph.ops.base import GraphTensor
from pimapper.core.graph.ops.torch_compat import create_torch_op_from_fx
from pimapper.modelmapper.passes.normalize_ops import NormalizeOpsPass
from pimapper.modelmapper.passes.simplify import SimplifyGraphPass


class NxGraphTracer(fx.Tracer):

    def is_leaf_module(self, m: Module, module_qualified_name: str) -> bool:
        if isinstance(m, RotaryPositionEmbedding):
            return True
        if isinstance(m, BatchedMatMulWithPast):
            return True
        return super().is_leaf_module(m, module_qualified_name) 


def _create_sample_inputs(module: torch.nn.Module, inference_config: InferenceConfig = None) -> tuple[torch.Tensor, ...]:
    """Create sample inputs for shape propagation."""
    if inference_config is None:
        inference_config = InferenceConfig()
    hidden_size = module.config.hidden_size
    # seq_len is always 1 for current token (not past_seq_len)
    return (torch.randn(inference_config.batch_size, 1, hidden_size),)


def trace_module(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[dict[str, Any]] = None,
    sample_inputs: Optional[tuple[torch.Tensor, ...]] = None,
    inference_config: Optional[InferenceConfig] = None,
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
        sample_inputs = _create_sample_inputs(root, inference_config)

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


def fx_to_computation_graph(graph: fx.Graph, module: Optional[torch.nn.Module] = None, inference_config: Optional[InferenceConfig] = None) -> NxComputationGraph:
    """Convert a torch.fx.Graph into a ComputationGraph using torch_compat ops.

    This function converts torch.fx graph nodes to torch_compat op representations,
    which can then be normalized to native ops via NormalizeOpsPass.

    Edges are automatically created by NxComputationGraph based on node references in args/kwargs.
    """
    comp_graph = NxComputationGraph()

    # Store inference_config in graph metadata
    if inference_config is not None:
        comp_graph.metadata['inference_config'] = {
            'batch_size': inference_config.batch_size,
            'past_seq_len': inference_config.past_seq_len,
            'data_format': inference_config.data_format  # Store DataFormat object
        }

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
            meta['module_class'] = target_module.__class__.__name__

            if isinstance(target_module, torch.nn.Linear):
                if hasattr(target_module, 'weight') and target_module.weight is not None:
                    meta['weight_shape'] = tuple(target_module.weight.shape)
                if hasattr(target_module, 'bias') and target_module.bias is not None:
                    meta['bias_shape'] = tuple(target_module.bias.shape)
                meta['in_features'] = target_module.in_features
                meta['out_features'] = target_module.out_features

            elif isinstance(target_module, BatchedMatMulWithPast):
                meta['batch_size'] = target_module.input_batch_size
                meta['num_matmuls'] = target_module.num_matmuls
                meta['matmul_shape'] = target_module.matmul_shape
                meta['is_qk_matmul'] = target_module.is_qk_matmul
                meta['model_config'] = {
                    'hidden_size': target_module.model_config.hidden_size,
                    'intermediate_size': target_module.model_config.intermediate_size,
                    'num_hidden_layers': target_module.model_config.num_hidden_layers,
                    'num_attention_heads': target_module.model_config.num_attention_heads,
                    'num_key_value_heads': target_module.model_config.num_key_value_heads,
                    'vocab_size': target_module.model_config.vocab_size,
                    'model_type': target_module.model_config.model_type
                }
                meta['inference_config'] = {
                    'batch_size': target_module.inference_config.batch_size,
                    'past_seq_len': target_module.inference_config.past_seq_len
                }

        # Convert fx.Node references to string names in args and kwargs
        converted_args = convert_node_refs(node.args)
        converted_kwargs = convert_node_refs(node.kwargs)

        # Create torch_compat Op from fx node with converted args/kwargs
        op = create_torch_op_from_fx(node.op, node.target, args=converted_args, kwargs=converted_kwargs, metadata={"shape": shape, "dtype": dtype, "custom": meta})

        # 设置 results，目前所有 op 都只有一个输出
        op.results = [GraphTensor(shape=shape, dtype=dtype)]

        # Create node in computation graph with Op object
        comp_graph.create_node(name=node.name, op=op)

    return comp_graph


def build_computation_graph(
    card_path: str | Path,
    *,
    inference_config: Optional[InferenceConfig] = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    normalize: bool = True,
    simplify: bool = True,
) -> tuple[fx.Graph, NxComputationGraph]:
    """Full pipeline: load config, init module, trace, convert, normalize, and simplify.

    Args:
        card_path: Path to model configuration JSON file
        inference_config: InferenceConfig with batch_size and past_seq_len (defaults to batch_size=1, past_seq_len=1024)
        dtype: Data type for sample inputs
        device: Device for sample inputs
        normalize: Whether to apply NormalizeOpsPass (convert torch ops to native ops)
        simplify: Whether to apply SimplifyGraphPass (remove non-essential ops)

    Returns:
        Tuple of (torch.fx.Graph, NxComputationGraph)
        The NxComputationGraph will have normalized and/or simplified ops based on parameters.
    """
    if inference_config is None:
        inference_config = InferenceConfig()

    config = load_model_config(card_path)
    module = initialize_module(config, inference_config=inference_config, dtype=dtype, device=device)

    hidden_size = config.hidden_size
    # seq_len is always 1 for current token (not past_seq_len)
    sample_input = torch.randn(inference_config.batch_size, 1, hidden_size, dtype=dtype, device=device)

    # Step 1: Trace the module to get torch.fx graph
    graph = trace_module(module, sample_inputs=(sample_input,), inference_config=inference_config)

    # Step 2: Convert to NxComputationGraph (with torch_compat ops)
    comp_graph = fx_to_computation_graph(graph, module, inference_config=inference_config)

    # Step 3: Apply NormalizeOpsPass (torch_compat -> native ops)
    if normalize:
        normalize_pass = NormalizeOpsPass()
        normalize_pass.run(comp_graph)

    # Step 4: Apply SimplifyGraphPass (remove non-essential ops)
    if simplify:
        simplify_pass = SimplifyGraphPass()
        simplify_pass.run(comp_graph)

    return graph, comp_graph








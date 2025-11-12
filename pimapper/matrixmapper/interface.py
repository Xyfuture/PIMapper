    from typing import Literal, Optional
from ..core.hwspec import AcceleratorSpec, Accelerator
from ..core.matrixspec import MatrixShape
from ..core.utils import MappingResult
from .strategy.trivial import TrivialTilingStrategy
from .strategy.recursive_grid_search import RecursiveGridSearchStrategy
from .strategy.h2llm_mapping import H2LLMTilingStrategy

StrategyType = Literal["trivial", "recursive_grid_search", "h2llm"]


def create_mapping(
    matrix_shape: MatrixShape,
    accelerator_spec: AcceleratorSpec,
    strategy: StrategyType = "trivial",
    **kwargs
) -> Optional[MappingResult]:
    """Create a mapping from matrix shape to hardware using the specified strategy.

    Args:
        matrix_shape: Matrix dimensions to map
        accelerator_spec: Hardware accelerator specification
        strategy: Mapping strategy to use ("trivial", "recursive_grid_search", "h2llm")
        **kwargs: Strategy-specific parameters

    Returns:
        MappingResult with mapping and latency, or None if mapping failed
    """
    accelerator = Accelerator.create_from_spec(accelerator_spec)

    if strategy == "trivial":
        strat = TrivialTilingStrategy()
        mapping = strat.create_balanced_mapping(matrix_shape, accelerator, **kwargs)
        from .evaluator import evaluate
        latency = evaluate(accelerator, mapping)
        return MappingResult(mapping=mapping, latency=latency)

    elif strategy == "recursive_grid_search":
        strat = RecursiveGridSearchStrategy(**kwargs)
        return strat.find_optimal_mapping(matrix_shape, accelerator)

    elif strategy == "h2llm":
        strat = H2LLMTilingStrategy(**kwargs)
        return strat.find_optimal_mapping(matrix_shape, accelerator)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

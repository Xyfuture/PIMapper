"""Vector operation latency calculation pass.

This pass calculates latency for vector operations in the computation graph
based on the host processor's vector compute power.
"""

import math
from typing import Optional

from pimapper.modelmapper.passes.base import Pass
from pimapper.core.graph.base import NxComputationGraph
from pimapper.core.hwspec import HostSpec


# FLOPs coefficient for each vector operation type
# These represent the approximate number of FLOPs per element
VECTOR_OP_FLOPS = {
    "vector_add": 1.0,
    "vector_mul": 1.0,
    "vector_dot": 2.0,  # Multiply + accumulate
    "silu": 4.0,  # x * sigmoid(x) involves exp, division, multiply
    "softmax": 6.0,  # exp, sum, division per element
    "rmsnorm": 5.0,  # square, mean, sqrt, division, multiply
}


class VectorLatencyPass(Pass):
    """Pass that calculates latency for vector operations.

    This pass:
    1. Traverses all nodes in the computation graph
    2. Identifies vector operations (vector_add, vector_mul, silu, softmax, rmsnorm, etc.)
    3. Extracts vector dimensions and batch information from op.results
    4. Calculates latency based on HostSpec's vector_compute_power
    5. Stores the latency in op.kwargs['latency']

    Args:
        host_spec: Host specification containing vector compute power
        name: Optional custom name for the pass
    """

    def __init__(
        self,
        host_spec: HostSpec,
        *,
        name: Optional[str] = None
    ):
        super().__init__(
            name=name or "VectorLatencyPass",
            description="Calculates latency for vector operations based on host compute power"
        )
        self.host_spec = host_spec

    def _extract_vector_info(self, op) -> Optional[tuple[int, int]]:
        """Extract vector dimensions from an operation.

        Args:
            op: Operation object to extract info from

        Returns:
            Tuple of (batch_size, vector_length) or None if extraction fails
        """
        # Check if op has results
        if not hasattr(op, 'results') or not op.results:
            return None

        # Get first result (output tensor)
        result = op.results[0]

        # Get shape
        shape = getattr(result, 'shape', None)
        if shape is None or len(shape) < 1:
            return None

        # Extract batch and vector dimensions
        if len(shape) == 1:
            # 1D vector: no batch dimension
            batch_size = 1
            vector_length = shape[0]
        else:
            # Multi-dimensional tensor: last dimension is vector length
            # All other dimensions are batch dimensions
            batch_dims = shape[:-1]
            vector_length = shape[-1]
            batch_size = math.prod(batch_dims)

        return batch_size, vector_length

    def run(self, graph: NxComputationGraph) -> bool:
        """Execute the vector latency calculation pass.

        Args:
            graph: Computation graph to process

        Returns:
            True if any vector operations were processed, False otherwise

        Raises:
            RuntimeError: If a vector operation cannot be processed
        """
        # Initialize metadata
        self._metadata = {
            "vectors_processed": 0,
            "total_latency": 0.0,
            "failed_operations": [],
            "latency_details": []
        }

        modified = False

        # Traverse all nodes in the graph
        for node_name in graph.nodes(sort=True):
            op = graph.node_record(node_name)

            # Check if this is a vector operation
            op_type = getattr(op, 'op_type', None)
            if op_type not in VECTOR_OP_FLOPS:
                continue

            # Extract vector information
            vector_info = self._extract_vector_info(op)
            if vector_info is None:
                # Skip if we can't extract vector info
                self._metadata["failed_operations"].append(node_name)
                continue

            batch_size, vector_length = vector_info

            # Get FLOPs coefficient for this operation
            flops_coefficient = VECTOR_OP_FLOPS[op_type]

            # Calculate total FLOPs
            total_flops = batch_size * vector_length * flops_coefficient

            # Calculate latency in seconds
            # host_spec.vector_compute_power is in GFLOPS (10^9 FLOPS)
            latency = total_flops / (self.host_spec.vector_compute_power ) # in ns with respect to matrix latency
 
            # Store latency in op.kwargs
            if not hasattr(op, 'kwargs'):
                op.kwargs = {}
            op.kwargs['latency'] = latency

            # Update metadata
            self._metadata["vectors_processed"] += 1
            self._metadata["total_latency"] += latency
            self._metadata["latency_details"].append({
                "node_name": node_name,
                "op_type": op_type,
                "shape": (batch_size, vector_length),
                "flops": total_flops,
                "latency": latency
            })

            modified = True

        return modified

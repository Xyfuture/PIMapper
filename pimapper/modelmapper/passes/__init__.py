"""Passes framework for computation graph optimization"""

from pimapper.modelmapper.passes.base import Pass, PassManager
from pimapper.modelmapper.passes.matrix_fusion import MatrixFusionPass
from pimapper.modelmapper.passes.simplify import SimplifyGraphPass

__all__ = ["Pass", "PassManager", "MatrixFusionPass", "SimplifyGraphPass"]

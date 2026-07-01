"""Processes subpackage: pluggable stochastic market-price dynamics."""

from bioventure.processes.base import StochasticProcess
from bioventure.processes.gbm import GeometricBrownianMotion
from bioventure.processes.jump_diffusion import JumpDiffusion
from bioventure.processes.regime_switching import RegimeSwitching
from bioventure.processes.registry import get_process, list_processes, register_process

__all__ = [
    "GeometricBrownianMotion",
    "JumpDiffusion",
    "RegimeSwitching",
    "StochasticProcess",
    "get_process",
    "list_processes",
    "register_process",
]

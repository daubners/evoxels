"""Built-in partial differential equation models."""

from .base import ODE, SemiLinearODE, SmoothedBoundaryODE
from .phase_field import (
    CahnHilliard,
    CahnHilliardSBM,
    MultiWellAllenCahn,
    TwoPhaseAllenCahn,
)
from .reaction_diffusion import (
    CoupledReactionDiffusion,
    ReactionDiffusion,
    ReactionDiffusionSBM,
)

__all__ = [
    "ODE",
    "CahnHilliard",
    "CahnHilliardSBM",
    "CoupledReactionDiffusion",
    "MultiWellAllenCahn",
    "ReactionDiffusion",
    "ReactionDiffusionSBM",
    "SemiLinearODE",
    "SmoothedBoundaryODE",
    "TwoPhaseAllenCahn",
]

"""Distillation module for magic state distillation protocols and overhead analysis."""

from .bravyi_kitaev import (
    BravyiKitaevDistillation,
    DistillationResult,
    DistillationCascade,
    compare_injection_protocols_distillation_cost
)

from .overhead_calculator import (
    OverheadCalculator,
    OverheadMetrics,
    SurfaceCodeParameters,
    DistillationFactoryConfig
)

__all__ = [
    'BravyiKitaevDistillation',
    'DistillationResult',
    'DistillationCascade',
    'compare_injection_protocols_distillation_cost',
    'OverheadCalculator',
    'OverheadMetrics',
    'SurfaceCodeParameters',
    'DistillationFactoryConfig'
]

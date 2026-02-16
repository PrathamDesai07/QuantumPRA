"""Manuscript preparation module for data aggregation and LaTeX generation."""

from .data_aggregator import (
    ManuscriptDataAggregator,
    ProtocolSummary,
    ComparisonMetrics
)

__all__ = [
    'ManuscriptDataAggregator',
    'ProtocolSummary',
    'ComparisonMetrics'
]

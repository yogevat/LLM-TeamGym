"""Topology subpackage — enumeration, geometric generation, spectral analysis, and learned search."""

from llm_team_gym.topology.enumeration import enumerate_shapes, count_shapes, count_all_shapes
from llm_team_gym.topology.geometric import GeometricGenerator, TopologyDescriptor
from llm_team_gym.topology.spectral import SpectralAnalyzer, TopologyClusterer, TopologyRanker
from llm_team_gym.topology.learned import TopologyController, TopologySearchLoop

__all__ = [
    # enumeration
    "enumerate_shapes",
    "count_shapes",
    "count_all_shapes",
    # geometric
    "GeometricGenerator",
    "TopologyDescriptor",
    # spectral
    "SpectralAnalyzer",
    "TopologyClusterer",
    "TopologyRanker",
    # learned
    "TopologyController",
    "TopologySearchLoop",
]

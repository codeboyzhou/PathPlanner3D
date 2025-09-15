from pp3d.algorithm.pso.types import PSOAlgorithmArguments


class DynamicWeightsPSOAlgorithmArguments(PSOAlgorithmArguments):
    """A class for dynamic weights PSO algorithm arguments."""

    cognitive_weight_min: float
    """The minimum cognitive weight."""

    cognitive_weight_max: float
    """The maximum cognitive weight."""

    social_weight_min: float
    """The minimum social weight."""

    social_weight_max: float
    """The maximum social weight."""

import numpy as np
from pydantic import BaseModel, ConfigDict


class AlgorithmIterationResult(BaseModel):
    """A class for algorithm iteration result."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    """Configuration for the model."""

    pso_full_path_points: np.ndarray
    """Full path points for PSO algorithm."""

    pso_best_fitness_values: list[float]
    """Best fitness values for PSO algorithm."""

    ga_full_path_points: np.ndarray
    """Full path points for GA algorithm."""

    ga_best_fitness_values: list[float]
    """Best fitness values for GA algorithm."""

    pso_ga_hybrid_full_path_points: np.ndarray
    """Full path points for PSO-GA hybrid algorithm."""

    pso_ga_hybrid_best_fitness_values: list[float]
    """Best fitness values for PSO-GA hybrid algorithm."""

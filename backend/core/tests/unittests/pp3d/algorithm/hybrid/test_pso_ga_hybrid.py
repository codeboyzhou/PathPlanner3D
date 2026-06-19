import numpy as np
from pp3d.algorithm.hybrid.pso_ga_hybrid import PSOGAHybridAlgorithm
from pp3d.algorithm.hybrid.types import HybridPSOAlgorithmArguments


def fitness_function(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def test_run() -> None:
    args = HybridPSOAlgorithmArguments(
        num_particles=8,
        num_waypoints=3,
        max_iterations=3,
        inertia_weight_min=0.4,
        inertia_weight_max=0.9,
        cognitive_weight_min=0.5,
        cognitive_weight_max=2.0,
        social_weight_min=0.5,
        social_weight_max=2.0,
        max_velocities=(1.0, 1.0, 1.0),
        axes_min=(0.0, 0.0, 0.0),
        axes_max=(10.0, 10.0, 10.0),
        random_seed=42,
    )

    planner = PSOGAHybridAlgorithm(args, fitness_function)
    path, convergence = planner.run()

    assert path.shape == (args.num_waypoints, 3)
    assert len(convergence) == args.max_iterations

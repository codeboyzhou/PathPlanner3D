from collections.abc import Callable

import numpy as np

import pytest
from pp3d.algorithm.pso.pso import PSO
from pp3d.algorithm.pso.types import PSOAlgorithmArguments
from pp3d.common.types import ProblemType


@pytest.fixture
def minimization_fitness_function() -> Callable[[np.ndarray], float]:
    """Define a simple minimization fitness function for testing."""
    return lambda x: np.sum(x**2)


@pytest.fixture
def maximization_fitness_function() -> Callable[[np.ndarray], float]:
    """Define a simple maximization fitness function for testing."""
    return lambda x: -np.sum(x**2)


@pytest.fixture
def algorithm_args():
    """Define algorithm arguments for testing."""
    return PSOAlgorithmArguments(
        num_particles=10,
        num_waypoints=5,
        max_iterations=10,
        inertia_weight_min=0.4,
        inertia_weight_max=0.9,
        cognitive_weight=1.5,
        social_weight=1.5,
        max_velocities=(1.0, 1.0, 1.0),
        axes_min=(-10.0, -10.0, -10.0),
        axes_max=(10.0, 10.0, 10.0),
        random_seed=42,
        verbose=True,
    )


def test_pso_initialization(algorithm_args, minimization_fitness_function):
    """Test PSO initialization with default minimization problem type."""
    pso = PSO(algorithm_args, minimization_fitness_function)

    # Check that the PSO object is correctly initialized
    assert pso.problem_type == ProblemType.MINIMIZATION
    assert len(pso.particles) == algorithm_args.num_particles
    assert pso.global_best_position is not None
    assert pso.global_best_fitness_value is not None


def test_pso_initialization_with_maximization(algorithm_args, minimization_fitness_function):
    """Test PSO initialization with maximization problem type."""
    pso = PSO(algorithm_args, minimization_fitness_function, ProblemType.MAXIMIZATION)

    # Check that the PSO object is correctly initialized with maximization
    assert pso.problem_type == ProblemType.MAXIMIZATION
    assert len(pso.particles) == algorithm_args.num_particles
    assert pso.global_best_position is not None
    assert pso.global_best_fitness_value is not None


def test_run_minimization(algorithm_args, minimization_fitness_function):
    """Test running PSO for a minimization problem."""
    pso = PSO(algorithm_args, minimization_fitness_function, ProblemType.MINIMIZATION)

    # Run the PSO algorithm
    best_path_points, best_fitness_values = pso.run()

    # Check the results
    assert best_path_points.shape == (algorithm_args.num_waypoints, 3)
    assert len(best_fitness_values) == algorithm_args.max_iterations

    # With a simple quadratic function, the best fitness should be non-negative and ideally close to zero
    assert best_fitness_values[-1] >= 0


def test_run_maximization(algorithm_args, maximization_fitness_function):
    """Test running PSO for a maximization problem."""
    pso = PSO(algorithm_args, maximization_fitness_function, ProblemType.MAXIMIZATION)

    # Run the PSO algorithm
    best_path_points, best_fitness_values = pso.run()

    # Check the results
    assert best_path_points.shape == (algorithm_args.num_waypoints, 3)
    assert len(best_fitness_values) == algorithm_args.max_iterations

    # With a negative quadratic function, the best fitness should be non-positive
    assert best_fitness_values[-1] <= 0

import time
from collections.abc import Callable

import numpy as np
import streamlit as st
from loguru import logger

from pp3d.algorithm.pso.pso import PSOAlgorithm
from pp3d.algorithm.pso.types import PSOAlgorithmArguments


def init_algorithm_args() -> PSOAlgorithmArguments:
    """Initialize PSO algorithm arguments for the 3D Path Planning Playground.

    Returns:
        PSOAlgorithmArguments: The initialized PSO algorithm arguments for the 3D Path Planning Playground.
    """
    with st.expander(label="PSO Arguments", expanded=True):
        num_particles = st.number_input("Number of Particles", min_value=10, max_value=100, value=50, step=10)
        num_waypoints = st.number_input("Number of Waypoints", min_value=2, max_value=50, value=4, step=1)
        max_iterations = st.number_input("Max Iterations", min_value=10, max_value=1000, value=100, step=10)
        inertia_weight = st.number_input("Inertia Weight", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
        cognitive_weight = st.number_input("Cognitive Weight", min_value=0.1, max_value=2.1, value=1.5, step=0.1)
        social_weight = st.number_input("Social Weight", min_value=0.1, max_value=2.1, value=1.5, step=0.1)
        with st.expander(label="Max Velocities", expanded=True):
            max_velocity_x = st.number_input("Max Velocity X", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            max_velocity_y = st.number_input("Max Velocity Y", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            max_velocity_z = st.number_input("Max Velocity Z", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        with st.expander(label="Axes Min", expanded=True):
            axes_min_x = st.number_input("Axis Min X", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
            axes_min_y = st.number_input("Axis Min Y", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
            axes_min_z = st.number_input("Axis Min Z", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        with st.expander(label="Axes Max", expanded=True):
            axes_max_x = st.number_input("Axis Max X", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
            axes_max_y = st.number_input("Axis Max Y", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
            axes_max_z = st.number_input("Axis Max Z", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
        random_seed = st.number_input(
            "Random Seed (0 means None, for non-deterministic)", min_value=0, max_value=1000, value=0, step=1
        )
        verbose = st.checkbox("Verbose")
        return PSOAlgorithmArguments(
            num_particles=num_particles,
            num_waypoints=num_waypoints,
            max_iterations=max_iterations,
            inertia_weight=inertia_weight,
            cognitive_weight=cognitive_weight,
            social_weight=social_weight,
            max_velocities=(max_velocity_x, max_velocity_y, max_velocity_z),
            axes_min=(axes_min_x, axes_min_y, axes_min_z),
            axes_max=(axes_max_x, axes_max_y, axes_max_z),
            random_seed=random_seed,
            verbose=verbose,
        )


def run_algorithm(
    args: PSOAlgorithmArguments, fitness_function: Callable[[np.ndarray], float]
) -> tuple[np.ndarray, list[float], float]:
    """Run the PSO algorithm for the 3D Path Planning Playground.

    Args:
        args (PSOAlgorithmArguments): The PSO algorithm arguments for the 3D Path Planning Playground.
        fitness_function (Callable[[np.ndarray], float]): The fitness function for the 3D Path Planning Playground.

    Returns:
        tuple[np.ndarray, list[float], float]: The best path points, best fitness values, and the time cost.
    """
    start_time = time.perf_counter()
    pso = PSOAlgorithm(args, fitness_function)
    best_path_points, best_fitness_values = pso.run()
    end_time = time.perf_counter()
    duration = end_time - start_time
    return best_path_points, best_fitness_values, duration


def run_algorithm_multiple_times(
    args: PSOAlgorithmArguments, fitness_function: Callable[[np.ndarray], float], times: int = 100
) -> tuple[list[float], list[float]]:
    """Run the PSO algorithm for the 3D Path Planning Playground multiple times.

    Args:
        args (PSOAlgorithmArguments): The PSO algorithm arguments for the 3D Path Planning Playground.
        fitness_function (Callable[[np.ndarray], float]): The fitness function for the 3D Path Planning Playground.
        times (int, optional): The number of times to run the PSO algorithm. Defaults to 100.

    Returns:
        tuple[list[float], list[float]]: The best fitness values for each time, and the time cost for each time.
    """
    best_fitness_list: list[float] = []
    duration_list: list[float] = []
    for loop in range(times):
        logger.info(f"Running PSO algorithm multiple times, current progress {loop + 1}/{times}.")
        _, best_fitness_values, duration = run_algorithm(args, fitness_function)
        best_fitness_list.append(best_fitness_values[-1])
        duration_list.append(duration)
    return best_fitness_list, duration_list

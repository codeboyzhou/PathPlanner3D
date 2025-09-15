from collections.abc import Callable

import numpy as np
import streamlit as st

from pp3d.algorithm.hybrid.dynamic_weights_pso import DynamicWeightsPSOAlgorithm
from pp3d.algorithm.hybrid.types import DynamicWeightsPSOAlgorithmArguments


def init_pso_algorithm_args() -> DynamicWeightsPSOAlgorithmArguments:
    """Initialize dynamic weights PSO algorithm arguments for the 3D Path Planning Playground.

    Returns:
        DynamicWeightsPSOAlgorithmArguments:
            The initialized dynamic weights PSO algorithm arguments for the 3D Path Planning Playground.
    """
    with st.expander(label="Dynamic Weights PSO Arguments", expanded=True):
        num_particles = st.number_input("Number of Particles", min_value=10, max_value=100, value=50, step=10)
        num_waypoints = st.number_input("Number of Waypoints", min_value=2, max_value=50, value=4, step=1)
        max_iterations = st.number_input("Max Iterations", min_value=10, max_value=1000, value=100, step=10)
        inertia_weight_min = st.number_input("Min Inertia Weight", min_value=0.1, max_value=2.0, value=0.4, step=0.1)
        inertia_weight_max = st.number_input("Max Inertia Weight", min_value=0.1, max_value=2.0, value=0.9, step=0.1)
        cognitive_weight_min = st.number_input(
            "Min Cognitive Weight", min_value=0.1, max_value=2.0, value=0.4, step=0.1
        )
        cognitive_weight_max = st.number_input(
            "Max Cognitive Weight", min_value=0.1, max_value=2.0, value=0.9, step=0.1
        )
        social_weight_min = st.number_input("Min Social Weight", min_value=0.1, max_value=2.0, value=0.4, step=0.1)
        social_weight_max = st.number_input("Max Social Weight", min_value=0.1, max_value=2.0, value=0.9, step=0.1)
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
        return DynamicWeightsPSOAlgorithmArguments(
            num_particles=num_particles,
            num_waypoints=num_waypoints,
            max_iterations=max_iterations,
            inertia_weight_min=inertia_weight_min,
            inertia_weight_max=inertia_weight_max,
            cognitive_weight_min=cognitive_weight_min,
            cognitive_weight_max=cognitive_weight_max,
            cognitive_weight=1.0,
            social_weight_min=social_weight_min,
            social_weight_max=social_weight_max,
            social_weight=1.0,
            max_velocities=(max_velocity_x, max_velocity_y, max_velocity_z),
            axes_min=(axes_min_x, axes_min_y, axes_min_z),
            axes_max=(axes_max_x, axes_max_y, axes_max_z),
            random_seed=random_seed,
            verbose=verbose,
        )


def run_pso_algorithm(
    args: DynamicWeightsPSOAlgorithmArguments, fitness_function: Callable[[np.ndarray], float]
) -> tuple[np.ndarray, list[float]]:
    """Run the dynamic weights PSO algorithm for the 3D Path Planning Playground.

    Args:
        args (DynamicWeightsPSOAlgorithmArguments):
            The dynamic weights PSO algorithm arguments for the 3D Path Planning Playground.
        fitness_function (Callable[[np.ndarray], float]): The fitness function for the 3D Path Planning Playground.

    Returns:
        tuple[np.ndarray, list[float]]: The best path points and best fitness values.
    """
    dynamic_weights_pso = DynamicWeightsPSOAlgorithm(args, fitness_function)
    best_path_points, best_fitness_values = dynamic_weights_pso.run()
    return best_path_points, best_fitness_values

from collections.abc import Callable

import numpy as np
import streamlit as st

from pp3d.algorithm.genetic.genetic import GeneticAlgorithm
from pp3d.algorithm.genetic.types import GeneticAlgorithmArguments


def init_genetic_algorithm_args() -> GeneticAlgorithmArguments:
    """Initialize genetic algorithm arguments for the 3D Path Planning Playground.

    Returns:
        GeneticAlgorithmArguments: The initialized genetic algorithm arguments for the 3D Path Planning Playground.
    """
    with st.expander(label="GA Arguments", expanded=True):
        population_size = st.number_input("Population Size", min_value=1, max_value=1000, value=100)
        num_waypoints = st.number_input("Number of Waypoints", min_value=5, max_value=100, value=5, step=5)
        max_iterations = st.number_input("Max Iterations", min_value=10, max_value=1000, value=100, step=10)
        crossover_rate = st.number_input("Crossover Rate", min_value=0.0, max_value=1.0, value=0.8)
        mutation_rate = st.number_input("Mutation Rate", min_value=0.0, max_value=1.0, value=0.1)
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
        return GeneticAlgorithmArguments(
            population_size=population_size,
            num_waypoints=num_waypoints,
            max_iterations=max_iterations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            axes_min=(axes_min_x, axes_min_y, axes_min_z),
            axes_max=(axes_max_x, axes_max_y, axes_max_z),
            random_seed=random_seed,
            verbose=verbose,
        )


def run_genetic_algorithm(
    args: GeneticAlgorithmArguments, fitness_function: Callable[[np.ndarray], float]
) -> tuple[np.ndarray, list[float]]:
    """Run the genetic algorithm for the 3D Path Planning Playground.

    Args:
        args (GeneticAlgorithmArguments): The genetic algorithm arguments for the 3D Path Planning Playground.
        fitness_function (Callable[[np.ndarray], float]): The fitness function for the 3D Path Planning Playground.

    Returns:
        tuple[np.ndarray, list[float]]: The best path and the fitness history of the genetic algorithm.
    """
    ga = GeneticAlgorithm(args, fitness_function)
    best_path_points, best_fitness_values = ga.run()
    return best_path_points, best_fitness_values

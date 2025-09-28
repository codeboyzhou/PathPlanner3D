import time
from collections.abc import Callable

import numpy as np
import streamlit as st
from loguru import logger

from pp3d.algorithm.genetic.genetic import GeneticAlgorithm
from pp3d.algorithm.genetic.types import GeneticAlgorithmArguments
from pp3d.algorithm.types import AlgorithmArguments


def init_algorithm_args(common_algorithm_args: AlgorithmArguments) -> GeneticAlgorithmArguments:
    """Initialize genetic algorithm arguments for the 3D Path Planning Playground.

    Args:
        common_algorithm_args (AlgorithmArguments): The common algorithm arguments.

    Returns:
        GeneticAlgorithmArguments: The initialized genetic algorithm arguments for the 3D Path Planning Playground.
    """
    with st.expander(label="GA Arguments", expanded=True):
        population_size = st.number_input("Population Size", min_value=1, max_value=1000, value=100, step=1)
        tournament_size = st.number_input("Tournament Size", min_value=2, max_value=10, value=3, step=1)
        crossover_rate = st.number_input("Crossover Rate", min_value=0.0, max_value=1.0, value=0.8)
        mutation_rate = st.number_input("Mutation Rate", min_value=0.0, max_value=1.0, value=0.2)
        return GeneticAlgorithmArguments(
            population_size=population_size,
            tournament_size=tournament_size,
            num_waypoints=common_algorithm_args.num_waypoints,
            max_iterations=common_algorithm_args.max_iterations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            axes_min=common_algorithm_args.axes_min,
            axes_max=common_algorithm_args.axes_max,
            random_seed=common_algorithm_args.random_seed,
            verbose=common_algorithm_args.verbose,
        )


def run_algorithm(
    args: GeneticAlgorithmArguments, fitness_function: Callable[[np.ndarray], float]
) -> tuple[np.ndarray, list[float], float]:
    """Run the genetic algorithm for the 3D Path Planning Playground.

    Args:
        args (GeneticAlgorithmArguments): The genetic algorithm arguments for the 3D Path Planning Playground.
        fitness_function (Callable[[np.ndarray], float]): The fitness function for the 3D Path Planning Playground.

    Returns:
        tuple[np.ndarray, list[float], float]: The best path points, best fitness values, and the time cost.
    """
    start_time = time.perf_counter()
    ga = GeneticAlgorithm(args, fitness_function)
    best_path_points, best_fitness_values = ga.run()
    end_time = time.perf_counter()
    duration = end_time - start_time
    return best_path_points, best_fitness_values, duration


def run_algorithm_multiple_times(
    args: GeneticAlgorithmArguments, fitness_function: Callable[[np.ndarray], float], times: int = 100
) -> tuple[list[float], list[float]]:
    """Run the genetic algorithm for the 3D Path Planning Playground multiple times.

    Args:
        args (GeneticAlgorithmArguments): The genetic algorithm arguments for the 3D Path Planning Playground.
        fitness_function (Callable[[np.ndarray], float]): The fitness function for the 3D Path Planning Playground.
        times (int, optional): The number of times to run the genetic algorithm. Defaults to 100.

    Returns:
        tuple[list[float], list[float]]: The best fitness values for each time, and the time cost for each time.
    """
    best_fitness_list: list[float] = []
    duration_list: list[float] = []
    for loop in range(times):
        logger.info(f"Running genetic algorithm multiple times, current progress {loop + 1}/{times}.")
        _, best_fitness_values, duration = run_algorithm(args, fitness_function)
        best_fitness_list.append(best_fitness_values[-1])
        duration_list.append(duration)
    return best_fitness_list, duration_list

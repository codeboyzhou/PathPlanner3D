from pp3d.common.types import ProblemType


def compare_fitness(fitness1: float, fitness2: float, problem_type: ProblemType) -> bool:
    """Compare two fitness values by the problem type.

    Args:
        fitness1 (float): The fitness value.
        fitness2 (float): The another fitness value.
        problem_type (ProblemType): The type of the problem.

    Returns:
        bool: True if fitness1 is better than fitness2, False otherwise.
    """
    return fitness1 < fitness2 if problem_type == ProblemType.MINIMIZATION else fitness1 > fitness2


def check_fitness_convergence(best_fitness_values: list[float], convergence_threshold: float = 1e-6) -> bool:
    """Check if the fitness values converge.

    Args:
        best_fitness_values (list[float]): The list of best fitness values.
        convergence_threshold (float, optional): The threshold of convergence. Defaults to 1e-6.

    Returns:
        bool: True if the fitness values converge, False otherwise.
    """
    if len(best_fitness_values) < 2:
        return False

    last_fitness = best_fitness_values[-1]
    second_last_fitness = best_fitness_values[-2]

    return abs(last_fitness - second_last_fitness) < convergence_threshold

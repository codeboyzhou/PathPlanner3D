from collections.abc import Callable
from typing import override

import numpy as np
from loguru import logger

from pp3d.algorithm.hybrid.types import DynamicWeightsPSOAlgorithmArguments
from pp3d.algorithm.pso.pso import PSOAlgorithm
from pp3d.common.types import ProblemType


class DynamicWeightsPSOAlgorithm(PSOAlgorithm):
    """A class for dynamic weights PSO algorithm."""

    @override
    def __init__(
        self,
        args: DynamicWeightsPSOAlgorithmArguments,
        fitness_function: Callable[[np.ndarray], float],
        problem_type: ProblemType = ProblemType.MINIMIZATION,
    ) -> None:
        """Initialize the dynamic weights PSO algorithm.

        Args:
            args (DynamicWeightsPSOAlgorithmArguments): The arguments for the dynamic weights PSO algorithm.
            fitness_function (Callable[[np.ndarray], float]): The fitness function.
            problem_type (ProblemType, optional): The type of the problem. Defaults to ProblemType.MINIMIZATION.
        """
        super().__init__(args, fitness_function, problem_type)
        self.args = args
        logger.success(f"Dynamic weights PSO algorithm initialized with {args}")

    @override
    def _get_inertia_weight(self, current_iteration: int) -> float:
        """Get the inertia weight for the current iteration.

        Args:
            current_iteration (int): The current iteration of the algorithm.

        Returns:
            float: The inertia weight for the current iteration.
        """
        inertia_weight = self.args.inertia_weight_min + (
            self.args.inertia_weight_max - self.args.inertia_weight_min
        ) * self._get_gaussian_perturbation(current_iteration)
        logger.debug(f"Iteration {current_iteration}/{self.args.max_iterations}, inertia_weight = {inertia_weight}")
        return inertia_weight

    @override
    def _get_cognitive_weight(self, current_iteration: int) -> float:
        """Get the cognitive weight for the current iteration.

        Args:
            current_iteration (int): The current iteration of the algorithm.

        Returns:
            float: The cognitive weight for the current iteration.
        """
        cognitive_weight = self.args.cognitive_weight_min + (
            self.args.cognitive_weight_max - self.args.cognitive_weight_min
        ) * self._get_gaussian_perturbation(current_iteration)
        logger.debug(f"Iteration {current_iteration}/{self.args.max_iterations}, cognitive_weight = {cognitive_weight}")
        return cognitive_weight

    @override
    def _get_social_weight(self, current_iteration: int) -> float:
        """Get the social weight for the current iteration.

        Args:
            current_iteration (int): The current iteration of the algorithm.

        Returns:
            float: The social weight for the current iteration.
        """
        social_weight = self.args.social_weight_min + (
            self.args.social_weight_max - self.args.social_weight_min
        ) * self._get_gaussian_perturbation(current_iteration)
        logger.debug(f"Iteration {current_iteration}/{self.args.max_iterations}, social_weight = {social_weight}")
        return social_weight

    def _get_gaussian_perturbation(self, current_iteration: int):
        """Get the gaussian perturbation for the current iteration.

        Args:
            current_iteration (int): The current iteration of the algorithm.

        Returns:
            float: The gaussian perturbation for the current iteration.
        """
        gaussian_perturbation = np.exp(-np.sin(2 * current_iteration**2 / self.args.max_iterations))
        logger.debug(
            f"Iteration {current_iteration}/{self.args.max_iterations}, "
            f"gaussian_perturbation = {gaussian_perturbation:.6f}"
        )
        return gaussian_perturbation

from collections.abc import Callable

import numpy as np
import streamlit as st
from loguru import logger
from scipy.ndimage import gaussian_filter
from streamlit_monaco_editor import st_monaco

from pp3d.algorithm.genetic.types import GeneticAlgorithmArguments
from pp3d.algorithm.hybrid.pso_types import DynamicPSOAlgorithmArguments
from pp3d.algorithm.pso.types import PSOAlgorithmArguments
from pp3d.common import collision_detection, interpolates
from pp3d.playground import (
    genetic_algorithm_playground,
    pso_algorithm_playground,
    pso_ga_hybrid_playground,
)
from pp3d.playground.constants import FITNESS_FUNCTION_CODE_TEMPLATE, TERRAIN_GENERATION_CODE_TEMPLATE
from pp3d.playground.types import AlgorithmIterationResult
from pp3d.visualization import plotly_utils


def _init_streamlit_session_state():
    """Initialize the session state of streamlit."""
    if "run_selected_algorithm" not in st.session_state:
        st.session_state.run_selected_algorithm = False
    if "run_multiple_algorithms" not in st.session_state:
        st.session_state.run_multiple_algorithms = False


def _st_monaco_editor(value: str, language: str = "python", height: str = "480px", theme: str = "vs-dark") -> str:
    """Create a Monaco editor widget in Streamlit.

    Args:
        value (str): The initial value of the editor.
        language (str): The programming language for syntax highlighting.
        height (str): The height of the editor.
        theme (str): The theme of the editor.

    Returns:
        str: The code written in the editor.
    """
    return st_monaco(value=value, language=language, height=height, theme=theme)


class Playground:
    """A class for the 3D Path Planning Playground."""

    def __init__(self):
        """Initialize the 3D Path Planning Playground."""
        st.set_page_config(page_title="3D Path Planning Playground", page_icon="🚢", layout="wide")
        self.left, self.middle, self.right = st.columns([2, 4, 4])

        self.selected_algorithm: str = "PSO"
        self.selected_algorithm_args: (
            PSOAlgorithmArguments | GeneticAlgorithmArguments | DynamicPSOAlgorithmArguments | None
        ) = None
        self.input_terrain_generation_code: str = TERRAIN_GENERATION_CODE_TEMPLATE
        self.input_fitness_function_code: str = FITNESS_FUNCTION_CODE_TEMPLATE

        _init_streamlit_session_state()
        self._init_left_column()
        self._init_middle_column()
        self._init_right_column()

    def _init_left_column(self) -> None:
        """Initialize the left column of the 3D Path Planning Playground."""
        with self.left:
            st.header("⚙️ Algorithm Settings")
            self.selected_algorithm = st.selectbox("Select Algorithm", ["PSO", "GA", "PSO-GA Hybrid"])
            if self.selected_algorithm == "PSO":
                self.selected_algorithm_args = pso_algorithm_playground.init_algorithm_args()
            elif self.selected_algorithm == "GA":
                self.selected_algorithm_args = genetic_algorithm_playground.init_algorithm_args()
            elif self.selected_algorithm == "PSO-GA Hybrid":
                self.selected_algorithm_args = pso_ga_hybrid_playground.init_algorithm_args()

    def _init_middle_column(self) -> None:
        """Initialize the middle column of the 3D Path Planning Playground."""
        with self.middle:
            st.header("💻 Code Editor")

            with st.expander(label="Terrain Generation", expanded=False):
                self.input_terrain_generation_code = _st_monaco_editor(value=TERRAIN_GENERATION_CODE_TEMPLATE)

            with st.expander(label="Fitness Function", expanded=False):
                self.input_fitness_function_code = _st_monaco_editor(value=FITNESS_FUNCTION_CODE_TEMPLATE)

            button_run_selected_algorithm_clicked = st.button(label="Run Selected Algorithm")
            if button_run_selected_algorithm_clicked:
                st.session_state.run_selected_algorithm = True

            button_run_multiple_algorithms_clicked = st.button(label="Run Multiple Algorithms")
            if button_run_multiple_algorithms_clicked:
                st.session_state.run_multiple_algorithms = True

    def _init_right_column(self) -> None:
        """Initialize the right column of the 3D Path Planning Playground."""
        with self.right:
            st.header("📊 Result Visualization")
            if st.session_state.run_selected_algorithm:
                self._run_selected_algorithm()
            if st.session_state.run_multiple_algorithms:
                self._run_multiple_algorithms()

    def _parse_fitness_function(
        self, start_point: np.ndarray, destination: np.ndarray, xx: np.ndarray, yy: np.ndarray, zz: np.ndarray
    ) -> Callable[[np.ndarray], float] | None:
        """Parse the input fitness function code to a function.

        Args:
            start_point (np.ndarray): The start point of the path.
            destination (np.ndarray): The destination point of the path.
            xx (np.ndarray): The x coordinates of the terrain height map.
            yy (np.ndarray): The y coordinates of the terrain height map.
            zz (np.ndarray): The z coordinates of the terrain height map.

        Returns:
            Callable[[np.ndarray], float] | None: The parsed fitness function.
        """
        try:
            allowed_packages = {
                "np": np,
                "xx": xx,
                "yy": yy,
                "zz": zz,
                "logger": logger,
                "interpolates": interpolates,
                "start_point": start_point,
                "destination": destination,
                "collision_detection": collision_detection,
            }
            parsed_fitness_function = {}
            exec(self.input_fitness_function_code, allowed_packages, parsed_fitness_function)
            callable_fitness_function = parsed_fitness_function["fitness_function"]
            return callable_fitness_function
        except Exception as e:
            st.error(f"Error parsing fitness function code: {e}")
            return None

    def _parse_terrain_generation_function(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray] | None:
        """Parse the input terrain generation function code to a function."""
        try:
            allowed_packages = {"np": np, "gaussian_filter": gaussian_filter}
            parsed_terrain_generation_function = {}
            exec(self.input_terrain_generation_code, allowed_packages, parsed_terrain_generation_function)
            callable_terrain_generation_function = parsed_terrain_generation_function["generate_terrain"]
            return callable_terrain_generation_function
        except Exception as e:
            st.error(f"Error parsing terrain generation code: {e}")

    def _generate_terrain(
        self, axes_min: tuple[float, float, float], axes_max: tuple[float, float, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the terrain using the input terrain generation function code.

        Args:
            axes_min (tuple[float, float, float]): The minimum values of the axes.
            axes_max (tuple[float, float, float]): The maximum values of the axes.

        Returns:
            xx (np.ndarray): The x-axis values of the terrain.
            yy (np.ndarray): The y-axis values of the terrain.
            zz (np.ndarray): The z-axis values of the terrain.
        """
        callable_terrain_generation_function = self._parse_terrain_generation_function()

        if callable_terrain_generation_function is None:
            st.error("Error running algorithm: callable_terrain_generation_function is None.")
            return np.array([]), np.array([]), np.array([])

        x = np.linspace(start=axes_min[0], stop=axes_max[0], num=100)
        y = np.linspace(start=axes_min[1], stop=axes_max[1], num=100)
        xx, yy = np.meshgrid(x, y)

        zz = callable_terrain_generation_function(xx, yy)

        return xx, yy, zz

    def _run_selected_algorithm(self) -> None:
        """Run the selected algorithm."""
        st.session_state.run_selected_algorithm = False

        if self.selected_algorithm_args is None:
            st.error("Error running algorithm: self.selected_algorithm_args is None.")
            return

        start_point = np.array([0, 0, 5])
        destination = np.array([90, 90, 5])

        axes_min = self.selected_algorithm_args.axes_min
        axes_max = self.selected_algorithm_args.axes_max
        xx, yy, zz = self._generate_terrain(axes_min, axes_max)

        callable_fitness_function = self._parse_fitness_function(start_point, destination, xx, yy, zz)

        if callable_fitness_function is None:
            st.error("Error running algorithm: callable_fitness_function is None.")
            return

        algorithm = self.selected_algorithm
        args = self.selected_algorithm_args
        best_path_points = np.array([])
        best_fitness_values = []

        if algorithm == "PSO" and isinstance(args, PSOAlgorithmArguments):
            best_path_points, best_fitness_values = pso_algorithm_playground.run_algorithm(
                args, callable_fitness_function
            )
        elif algorithm == "GA" and isinstance(args, GeneticAlgorithmArguments):
            best_path_points, best_fitness_values = genetic_algorithm_playground.run_algorithm(
                args, callable_fitness_function
            )
        elif algorithm == "PSO-GA Hybrid" and isinstance(args, DynamicPSOAlgorithmArguments):
            best_path_points, best_fitness_values = pso_ga_hybrid_playground.run_algorithm(
                args, callable_fitness_function
            )

        full_path_points = np.vstack([start_point, best_path_points, destination])

        plotly_utils.plot_terrain_and_path(xx, yy, zz, start_point, destination, full_path_points)
        plotly_utils.plot_fitness_curve(best_fitness_values)

    def _run_multiple_algorithms(self) -> None:
        """Run multiple algorithms."""
        st.session_state.run_multiple_algorithms = False

        start_point = np.array([0, 0, 5])
        destination = np.array([90, 90, 5])

        num_particles = 50
        num_waypoints = 4
        max_iterations = 300
        axes_min = (0, 0, 5)
        axes_max = (100, 100, 100)
        max_velocities = (1.0, 1.0, 1.0)

        pso_algorithm_args = PSOAlgorithmArguments(
            num_particles=num_particles,
            num_waypoints=num_waypoints,
            max_iterations=max_iterations,
            inertia_weight_min=0.4,
            inertia_weight_max=0.9,
            cognitive_weight=1.5,
            social_weight=1.5,
            max_velocities=max_velocities,
            axes_min=axes_min,
            axes_max=axes_max,
            random_seed=None,
            verbose=False,
        )
        ga_algorithm_args = GeneticAlgorithmArguments(
            population_size=num_particles,
            tournament_size=3,
            num_waypoints=num_waypoints,
            max_iterations=max_iterations,
            crossover_rate=0.8,
            mutation_rate=0.2,
            axes_min=axes_min,
            axes_max=axes_max,
            random_seed=None,
            verbose=False,
        )
        pso_ga_hybrid_algorithm_args = DynamicPSOAlgorithmArguments(
            num_particles=num_particles,
            num_waypoints=num_waypoints,
            max_iterations=max_iterations,
            inertia_weight_min=0.4,
            inertia_weight_max=0.9,
            cognitive_weight_min=0.5,
            cognitive_weight_max=2.5,
            social_weight_min=0.5,
            social_weight_max=2.5,
            max_velocities=max_velocities,
            axes_min=axes_min,
            axes_max=axes_max,
            random_seed=None,
            verbose=False,
        )

        # All algorithms use the same terrain, so we only need to generate the terrain once.
        xx, yy, zz = self._generate_terrain(pso_algorithm_args.axes_min, pso_algorithm_args.axes_max)

        callable_fitness_function = self._parse_fitness_function(start_point, destination, xx, yy, zz)

        if callable_fitness_function is None:
            st.error("Error running algorithm: callable_fitness_function is None.")
            return

        pso_path_points, pso_fitness_values = pso_algorithm_playground.run_algorithm(
            pso_algorithm_args, callable_fitness_function
        )
        ga_path_points, ga_fitness_values = genetic_algorithm_playground.run_algorithm(
            ga_algorithm_args, callable_fitness_function
        )
        pso_ga_hybrid_path_points, pso_ga_hybrid_fitness_values = pso_ga_hybrid_playground.run_algorithm(
            pso_ga_hybrid_algorithm_args, callable_fitness_function
        )

        algorithm_iteration_result = AlgorithmIterationResult(
            pso_full_path_points=np.vstack([start_point, pso_path_points, destination]),
            ga_full_path_points=np.vstack([start_point, ga_path_points, destination]),
            pso_ga_hybrid_full_path_points=np.vstack([start_point, pso_ga_hybrid_path_points, destination]),
            pso_best_fitness_values=pso_fitness_values,
            ga_best_fitness_values=ga_fitness_values,
            pso_ga_hybrid_best_fitness_values=pso_ga_hybrid_fitness_values,
        )

        plotly_utils.plot_terrain_and_multipath(xx, yy, zz, start_point, destination, algorithm_iteration_result)
        plotly_utils.plot_multiple_fitness_curves(algorithm_iteration_result)


if __name__ == "__main__":
    Playground()

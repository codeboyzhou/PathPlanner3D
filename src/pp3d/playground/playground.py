from collections.abc import Callable

import numpy as np
import streamlit as st
from streamlit_monaco_editor import st_monaco

from pp3d.algorithm.pso.types import PSOAlgorithmArguments
from pp3d.playground import pso_playground
from pp3d.playground.constants import FITNESS_FUNCTION_CODE_TEMPLATE, TERRAIN_GENERATION_CODE_TEMPLATE
from pp3d.visualization import plotly_utils


def _init_streamlit_session_state():
    """Initialize the session state of streamlit."""
    if "terrain_figure_show" not in st.session_state:
        st.session_state.terrain_figure_show = False
    if "fitness_curve_figure_show" not in st.session_state:
        st.session_state.fitness_curve_figure_show = False


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
        self.selected_algorithm_args: PSOAlgorithmArguments | None = None
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
            self.selected_algorithm = st.selectbox("Select Algorithm", ["PSO"])
            if self.selected_algorithm == "PSO":
                self.selected_algorithm_args = pso_playground.init_pso_algorithm_args()

    def _init_middle_column(self) -> None:
        """Initialize the middle column of the 3D Path Planning Playground."""
        with self.middle:
            st.header("💻 Code Editor")

            with st.expander(label="Terrain Generation", expanded=False):
                self.input_terrain_generation_code = _st_monaco_editor(value=TERRAIN_GENERATION_CODE_TEMPLATE)

            button_generate_terrain_clicked = st.button(label="Generate Terrain")
            if button_generate_terrain_clicked:
                st.session_state.terrain_figure_show = True

            with st.expander(label="Fitness Function", expanded=False):
                self.input_fitness_function_code = _st_monaco_editor(value=FITNESS_FUNCTION_CODE_TEMPLATE)

            button_run_algorithm_clicked = st.button(label="Run Algorithm")
            if button_run_algorithm_clicked:
                st.session_state.fitness_curve_figure_show = True

    def _init_right_column(self) -> None:
        """Initialize the right column of the 3D Path Planning Playground."""
        with self.right:
            st.header("📊 Result Visualization")
            if st.session_state.terrain_figure_show:
                self._generate_terrain()
            if st.session_state.fitness_curve_figure_show:
                self._run_algorithm()

    def _parse_fitness_function(self) -> Callable[[np.ndarray], float] | None:
        """Parse the input fitness function code to a function."""
        try:
            allowed_packages = {"np": np}
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
            allowed_packages = {"np": np}
            parsed_terrain_generation_function = {}
            exec(self.input_terrain_generation_code, allowed_packages, parsed_terrain_generation_function)
            callable_terrain_generation_function = parsed_terrain_generation_function["generate_terrain"]
            return callable_terrain_generation_function
        except Exception as e:
            st.error(f"Error parsing terrain generation code: {e}")

    def _generate_terrain(self) -> None:
        """Generate the terrain."""
        callable_terrain_generation_function = self._parse_terrain_generation_function()

        if callable_terrain_generation_function is None:
            st.error("Error calling terrain generation function: callable_terrain_generation_function is None.")
            return

        axes_min = (0, 0, 0) if self.selected_algorithm_args is None else self.selected_algorithm_args.axes_min
        axes_max = (100, 100, 100) if self.selected_algorithm_args is None else self.selected_algorithm_args.axes_max
        x = np.linspace(axes_min[0], axes_max[0], 100)
        y = np.linspace(axes_min[1], axes_max[1], 100)
        xx, yy = np.meshgrid(x, y)

        zz = callable_terrain_generation_function(xx, yy)
        fig = plotly_utils.get_terrain_figure(xx, yy, zz)
        st.plotly_chart(fig, use_container_width=True)

    def _run_algorithm(self) -> None:
        """Run the selected algorithm."""
        callable_fitness_function = self._parse_fitness_function()

        if callable_fitness_function is None:
            st.error("Error calling fitness function: callable_fitness_function is None.")
            return

        if self.selected_algorithm_args is None:
            st.error("Error running algorithm: self.selected_algorithm_args is None.")
            return

        best_path_points = np.array([])
        best_fitness_values = []

        if self.selected_algorithm == "PSO":
            best_path_points, best_fitness_values = pso_playground.run_pso_algorithm(
                self.selected_algorithm_args, callable_fitness_function
            )

        fig = plotly_utils.get_fitness_curve_figure(best_fitness_values)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    Playground()

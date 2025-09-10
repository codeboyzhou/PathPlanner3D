import streamlit as st

from pp3d.playground import pso_playground


class Playground:
    """A class for the 3D Path Planning Playground."""

    def __init__(self):
        """Initialize the 3D Path Planning Playground."""
        st.set_page_config(page_title="3D Path Planning Playground", page_icon="🚢", layout="wide")
        self.left, self.right = st.columns([1, 5])

        self.selected_algorithm = None
        self.pso_args = None

        self._init_left_column()
        self._init_right_column()

    def _init_left_column(self):
        """Initialize the left column of the 3D Path Planning Playground."""
        with self.left:
            st.header("⚙️ Algorithm Settings")
            self.selected_algorithm = st.selectbox("Select Algorithm", ["PSO"])
            if self.selected_algorithm == "PSO":
                self.pso_args = pso_playground.init_pso_args()

    def _init_right_column(self):
        """Initialize the right column of the 3D Path Planning Playground."""
        with self.right:
            st.header("🚀 Planning Results")


if __name__ == "__main__":
    """Run the 3D Path Planning Playground."""
    Playground()

import streamlit as st
from streamlit_ace import st_ace

from pp3d.playground import i18n


def code_editor(
    value: str, language: str = "python", height: int = 480, theme: str = "dracula", font_size: int = 14
) -> str:
    """Create a code editor widget in Streamlit.

    Args:
        value (str): The initial value of the editor.
        language (str): The programming language for syntax highlighting. Defaults to "python".
        height (int): The height of the editor. Defaults to 480.
        theme (str): The theme of the editor. Defaults to "dracula".
        font_size (int): The font size of the editor. Defaults to 14.

    Returns:
        str: The code written in the editor.
    """
    return st_ace(value=value, language=language, height=height, theme=theme, font_size=font_size)


def select_language() -> str:
    """
    Create a language selection widget in Streamlit.

    Returns:
        str: The selected language.
    """
    return st.selectbox(
        width=200,
        label="Select Language / 选择语言",
        help="Select the language you are familiar with / 选择你熟悉的语言",
        options=list(i18n.translation.keys()),
        format_func=lambda selected: i18n.language_names[selected],
    )


def select_algorithm() -> str:
    """
    Create an algorithm selection widget in Streamlit.

    Returns:
        str: The selected algorithm.
    """
    return st.selectbox(
        label=i18n.translate("select_algorithm"),
        options=list(i18n.algorithm_names.keys()),
        format_func=lambda selected: i18n.algorithm_names[selected][st.session_state.selected_language],
    )


def input_multiple_runs() -> int:
    """
    Create a multiple runs number input widget in Streamlit.

    Returns:
        int: The input number of multiple runs.
    """
    return st.number_input(
        label=i18n.translate("multiple_runs"),
        help=i18n.translate("multiple_runs_help"),
        min_value=1,
        max_value=1000,
        value=100,
        step=100,
    )


def input_starting_point_coordinate() -> tuple[int, int, int]:
    """
    Create a starting point coordinate input widget in Streamlit.

    Returns:
        tuple[int, int, int]: The input coordinate of starting point.
    """
    with st.expander(label=i18n.translate("starting_point_coordinate"), expanded=True):
        x = st.number_input(
            label=i18n.translate("starting_point_coordinate_x"), min_value=0, max_value=1000, value=0, step=1
        )
        y = st.number_input(
            label=i18n.translate("starting_point_coordinate_y"), min_value=0, max_value=1000, value=0, step=1
        )
        z = st.number_input(
            label=i18n.translate("starting_point_coordinate_z"), min_value=0, max_value=1000, value=0, step=1
        )
        return x, y, z


def input_ending_point_coordinate() -> tuple[int, int, int]:
    """
    Create an ending point coordinate input widget in Streamlit.

    Returns:
        tuple[int, int, int]: The input coordinate of ending point.
    """
    with st.expander(label=i18n.translate("ending_point_coordinate"), expanded=True):
        x = st.number_input(
            label=i18n.translate("ending_point_coordinate_x"), min_value=0, max_value=1000, value=100, step=1
        )
        y = st.number_input(
            label=i18n.translate("ending_point_coordinate_y"), min_value=0, max_value=1000, value=100, step=1
        )
        z = st.number_input(
            label=i18n.translate("ending_point_coordinate_z"), min_value=0, max_value=1000, value=10, step=1
        )
        return x, y, z

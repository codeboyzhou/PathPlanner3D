from streamlit_ace import st_ace


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

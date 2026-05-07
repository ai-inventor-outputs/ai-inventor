"""User folder prompt component.

Provides user folder path context for agent and LLM prompts.
"""


def get_user_folder_prompt(user_folder_path: str) -> str:
    """Get user folder prompt with absolute path.

    Args:
        user_folder_path: Absolute path to the user's data folder.
    """
    return f"""<user_data>
User-provided reference materials are available at `{user_folder_path}`. Check this folder for anything relevant to your task.
</user_data>"""

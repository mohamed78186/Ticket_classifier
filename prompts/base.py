"""Shared prompt template used by every prompt-builder in this package.

Keeping one function in charge of the final formatting means every
prompt in the project has the same four-section shape (Role /
Constraints / Task / Input), and a formatting tweak only has to
happen in one place.
"""

_TEMPLATE = "Role:\n{role}\n\nConstraints:\n{constraints}\n\nTask:\n{task}\n\nInput:\n<input>\n{user_input}\n</input>"


def build_prompt(
    role: str,
    task: str,
    user_input: str,
    constraints: list[str] | None = None,
) -> str:
    """Assemble a single prompt string from its component parts."""
    bullet_points = [f"- {rule}" for rule in (constraints or [])]
    return _TEMPLATE.format(
        role=role,
        constraints="\n".join(bullet_points),
        task=task,
        user_input=user_input,
    )

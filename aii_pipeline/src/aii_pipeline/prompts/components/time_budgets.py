"""Time budget prompt component for planners (gen_hypo, gen_strat, gen_plan).

Shows artifact execution time limits so planners can propose feasible work.
Reads from the ambient PipelineConfig contextvar.
"""

from __future__ import annotations


def _fmt_minutes(seconds: int | None) -> str:
    if not seconds:
        return "?"
    m = seconds // 60
    if m >= 60:
        h = m // 60
        rm = m % 60
        return f"{h}h{rm}m" if rm else f"{h}h"
    return f"{m}m"


def get_time_budgets_overview() -> str:
    """All artifact type time budgets — for gen_hypo and gen_strat prompts."""
    from aii_pipeline.utils.context import get_pipeline_config

    config = get_pipeline_config()
    if not config:
        return ""

    exe = config.invention_loop.execute
    lines = [
        "<time_budgets>",
        "",
        "Each artifact executor has a fixed time budget (including writing code, debugging, testing, and fixing errors):",
        "",
    ]
    type_cfgs = [
        ("research", exe.research),
        ("dataset", exe.dataset),
        ("experiment", exe.experiment),
        ("evaluation", exe.evaluation),
        ("proof", exe.proof),
    ]
    for name, cfg in type_cfgs:
        timeout = cfg.claude_agent.agent_timeout if hasattr(cfg, "claude_agent") else None
        lines.append(f"- {name}: {_fmt_minutes(timeout)}")

    lines += [
        "",
        "</time_budgets>",
    ]
    return "\n".join(lines)


def get_time_budget_for_type(artifact_type: str) -> str:
    """Single artifact type time budget — for gen_plan prompts."""
    from aii_pipeline.utils.context import get_pipeline_config

    config = get_pipeline_config()
    if not config:
        return ""

    exe = config.invention_loop.execute
    cfg = getattr(exe, artifact_type, None)
    if not cfg:
        return ""

    timeout = cfg.claude_agent.agent_timeout if hasattr(cfg, "claude_agent") else None
    if not timeout:
        return ""

    return (
        f"<time_budget>\n\n"
        f"The {artifact_type} executor has {_fmt_minutes(timeout)} total "
        f"(including writing code, debugging, testing, and fixing errors).\n\n"
        f"</time_budget>"
    )

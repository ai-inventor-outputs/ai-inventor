"""Exec envs — where and how agent work runs.

Two-level resolution:
1. Env (where): ``LocalEnv``, ``RunPodEnv``
2. ``ComputeProfile`` (what resources): local system, GPU pod, CPU pod, etc.

Construct directly:
    env = LocalEnv()
    agent, result = await env.run_agent(options, prompts)
"""

from .base import ComputeProfile, ExecuteEnv
from .local import LocalEnv

__all__ = [
    "ComputeProfile",
    "ExecuteEnv",
    "LocalEnv",
]

"""Reusable Run-bus topology workflows.

Each subpackage is a self-contained pattern for wiring Runs together:

  - ``agent_worker`` — "run an agent, stream its events back" workflow.
    Has a local channel (in-process, events flow via current_run) and
    a RunPod channel (ephemeral worker pod, events streamed back via
    the sibling SSESink/SSESource pair).
"""

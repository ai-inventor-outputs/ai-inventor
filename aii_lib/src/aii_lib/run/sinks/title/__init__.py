"""aii_lib.run.sinks.title — derive a short run name from the first user prompt.

Subscribes to the Run bus. On the first ``AgentUserPromptMessage`` the
sink sees (when ``run.name`` is still empty), it fires
:func:`generate_title` on a daemon thread and emits a
:class:`RunTitleMessage` whose ``text`` is the LLM-generated title. The
dispatcher copies that text onto ``run.name``; every other sink picks
the name up via the same event. Subsequent prompts no-op.

The public helper :func:`generate_title` is also imported by the
aii_server start-run endpoint, which pre-computes the title before
spawning the pipeline subprocess so the dashboard sees a real name
within milliseconds instead of waiting for the bus to wake up.
"""

from __future__ import annotations

from .sink import TitleGeneratorSink, generate_title

__all__ = ["TitleGeneratorSink", "generate_title"]

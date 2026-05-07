"""Tree walker for per-node metric registration.

The metric pipeline (in :mod:`.provider`) walks the live :class:`Run`
tree and registers one :class:`ObservableCounter` per (node, stats field)
pair. The metric *name* itself encodes the leaf node identity
(``aii.{display_name}_{node_id}.{metric_suffix}``) so each series is
uniquely identified by name alone — no attribute pivot required in
Grafana.

In addition to the name, every emitted Observation carries its
**ancestor identity** as labels:

  * Run instrument       → ``{}`` (no ancestors)
  * Group instrument     → ``{aii.run_id}``
  * Iteration instrument → ``{aii.run_id, aii.group_id, aii.group}``
  * Module instrument    → ``{aii.run_id, aii.group_id, aii.group,
    aii.iteration?, aii.iter_id?}``

(``aii.iteration`` / ``aii.iter_id`` are included only when the module
is inside a ``LoopIteration`` — for sequential ``SeqMdGroup`` modules
they're omitted.)

Ancestor labels are crash-safe by the same logic as span lineage:
even if the run never ends cleanly, every metric data point carries its
ancestor identity, so queries like
``{aii_run_id="X", aii_module="gen_hypo"}`` work over partial state.
The leaf node's own identity stays in the metric name, not the labels,
to keep cardinality bounded.

Tasks are visited but filtered out by the provider — task series add
cardinality without signal (every task value rolls up into its parent
module). Per-task latency lives in the ``aii.task.duration`` histogram.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from aii_lib.run.run import Run


def _walk_nodes(
    run: Run,
) -> Iterable[tuple[str, object, str, dict[str, str]]]:
    """Yield structural node info for every node in the tree.

    Yield ``(level, node, display_name, ancestor_attrs)`` for every
    Run / Group / Iteration / Module / Task in the tree. ``display_name``
    is the friendly identifier baked into the metric name prefix —
    usually ``node.name``, but synthesized as ``iter_{N}`` for
    ``LoopIteration`` (which has an empty ``name`` field; its 1-based
    position in the parent group is the natural label).

    Ancestor attrs hold the IDs/names of all open ancestors of the
    yielded node (excluding the node's own identity).
    """
    from aii_lib.run.loop_iteration import LoopIteration
    from aii_lib.run.mdgroup import MdGroup
    from aii_lib.run.module import Module

    run_id = run.node_id or ""
    # Run is its own root scope — no ancestors above it.
    yield "run", run, run.name or "", {}

    for grp in list(run.children or []):
        if not isinstance(grp, MdGroup):
            continue
        group_id = grp.node_id or ""
        group_name = grp.name or ""
        yield "group", grp, group_name, {"aii.run_id": run_id}

        # Each group's children are either LoopIterations (for LoopMdGroup)
        # or Modules directly (for SeqMdGroup).
        group_ancestors = {
            "aii.run_id": run_id,
            "aii.group_id": group_id,
            "aii.group": group_name,
        }
        for idx, child in enumerate(list(grp.children or []), start=1):
            if isinstance(child, LoopIteration):
                yield (
                    "iteration",
                    child,
                    f"iter_{idx}",
                    dict(group_ancestors),
                )
                # Modules inside this iteration carry the iteration index
                # AND its node_id as ancestors so iter-level metrics can
                # join cleanly to module-level metrics by id.
                module_ancestors = dict(group_ancestors)
                module_ancestors["aii.iteration"] = idx
                module_ancestors["aii.iter_id"] = child.node_id or ""
                for mod in list(child.children or []):
                    if isinstance(mod, Module):
                        yield from _yield_module(mod, module_ancestors)
            elif isinstance(child, Module):
                yield from _yield_module(child, dict(group_ancestors))


def _yield_module(
    mod: object,
    ancestor_attrs: dict,
) -> Iterable[tuple[str, object, str, dict[str, str]]]:
    """Yield observations for one Module + all its Tasks, threading the

    module's identity through to its tasks as ancestor attrs.
    """
    from aii_lib.run.task import Task

    yield "module", mod, mod.name or "", dict(ancestor_attrs)

    task_ancestors = dict(ancestor_attrs)
    task_ancestors["aii.module_id"] = mod.node_id or ""
    task_ancestors["aii.module"] = mod.name or ""

    for task in list(mod.children or []):
        if isinstance(task, Task):
            yield "task", task, task.name or "", dict(task_ancestors)


__all__ = ["_walk_nodes"]

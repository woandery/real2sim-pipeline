"""Errors that stop a stage without silently qualifying incomplete work."""


class WorkflowError(RuntimeError):
    """A stage cannot produce trustworthy output."""


class IntegrityError(WorkflowError):
    """An input or completed artifact changed after it was recorded."""


class NeedsAttention(WorkflowError):
    """Evidence is insufficient or a bounded repair budget was exhausted."""

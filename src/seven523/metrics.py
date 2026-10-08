"""Training metrics sinks: the update CSV and its TensorBoard mirror.

Moved out of :mod:`seven523.train` so the trainer keeps only the rollout /
update loop.  The CSV header and row format are part of the run artefacts and
must stay stable (tests and :mod:`tools.backfill_tensorboard` read them).
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["LOG_FIELDS", "TB_TAGS", "MetricsLogger", "TensorboardLogger"]

#: Columns of ``metrics.csv``, in order; the header is ``global_step,<these>``.
LOG_FIELDS = (
    "episodic_return",
    "episodic_length",
    "episodes",
    "learning_rate",
    "value_loss",
    "policy_loss",
    "entropy",
    "old_approx_kl",
    "approx_kl",
    "clipfrac",
    "explained_variance",
    "outcome_loss",
    "outcome_accuracy",
    "sps",
    "eval_return",
    "eval_score",
    "eval_score_diff",
    "eval_win_rate",
)


class MetricsLogger:
    """Append one CSV row per update; no tensorboard/wandb dependency."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.write_text("global_step," + ",".join(LOG_FIELDS) + "\n")

    def log(self, global_step: int, **values: float | None) -> None:
        cells = []
        for field in LOG_FIELDS:
            value = values.get(field)
            cells.append("" if value is None else f"{value:.6g}")
        with self.path.open("a") as handle:
            handle.write(",".join([str(global_step), *cells]) + "\n")


#: CSV field -> TensorBoard tag for the scalars in :data:`LOG_FIELDS`.
TB_TAGS: dict[str, str] = {
    "episodes": "charts/episodes",
    "learning_rate": "charts/learning_rate",
    "value_loss": "losses/value_loss",
    "policy_loss": "losses/policy_loss",
    "entropy": "losses/entropy",
    "old_approx_kl": "losses/old_approx_kl",
    "approx_kl": "losses/approx_kl",
    "clipfrac": "losses/clipfrac",
    "explained_variance": "losses/explained_variance",
    "outcome_loss": "losses/outcome_loss",
    "outcome_accuracy": "losses/outcome_accuracy",
    "sps": "charts/SPS",
    "eval_return": "eval/mean_return",
    "eval_score": "eval/learner_score",
    "eval_score_diff": "eval/score_diff",
    "eval_win_rate": "eval/win_rate",
}


class TensorboardLogger:
    """Thin tensorboardX wrapper; no-op when disabled or not installed."""

    def __init__(self, log_dir: str | Path, enabled: bool = True) -> None:
        self.writer = None
        if enabled:
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                print(
                    "tensorboardX not installed; skipping TensorBoard "
                    "(metrics.csv is still written; run `uv sync --group train`)"
                )
            else:
                self.writer = SummaryWriter(str(log_dir))

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def add_scalar(self, tag: str, value: float | None, step: int) -> None:
        if self.writer is not None and value is not None:
            self.writer.add_scalar(tag, value, step)

    def add_text(self, tag: str, text: str, step: int = 0) -> None:
        if self.writer is not None:
            self.writer.add_text(tag, text, step)

    def log_update(self, global_step: int, metrics: dict[str, float | None]) -> None:
        for key, tag in TB_TAGS.items():
            self.add_scalar(tag, metrics.get(key), global_step)

    def log_episode(self, global_step: int, episode_return: float, length: int) -> None:
        self.add_scalar("charts/episodic_return", episode_return, global_step)
        self.add_scalar("charts/episodic_length", float(length), global_step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

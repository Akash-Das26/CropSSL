"""
Logging utilities for CropSSL.

Supports TensorBoard logging and structured file logging.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentLogger:
    """Structured experiment logging.

    Args:
        log_dir: Directory for log files.
        experiment_name: Name of the experiment.
        use_tensorboard: Whether to use TensorBoard.
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        experiment_name: str = "cropssl",
        use_tensorboard: bool = True,
    ):
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.log_dir / "tensorboard"))
            except ImportError:
                print("TensorBoard not available. Skipping.")

        # JSON log file
        self.log_file = self.log_dir / "metrics.jsonl"
        self.step = 0

    def log_scalar(
        self, tag: str, value: float, step: Optional[int] = None
    ):
        """Log a scalar value."""
        step = step if step is not None else self.step

        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

        # Append to JSONL
        entry = {"step": step, "tag": tag, "value": value}
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: Optional[int] = None,
    ):
        """Log multiple scalars under one main tag."""
        step = step if step is not None else self.step

        if self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        config_file = self.log_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

    def log_text(self, text: str, tag: str = "text"):
        """Log text."""
        if self.writer is not None:
            self.writer.add_text(tag, text, self.step)

    def increment_step(self):
        self.step += 1

    def close(self):
        """Close all loggers."""
        if self.writer is not None:
            self.writer.close()


class Timer:
    """Simple timer for tracking training stages."""

    def __init__(self):
        self.times: Dict[str, float] = {}
        self.starts: Dict[str, float] = {}

    def start(self, name: str):
        self.starts[name] = time.time()

    def stop(self, name: str) -> float:
        if name in self.starts:
            elapsed = time.time() - self.starts[name]
            self.times[name] = self.times.get(name, 0) + elapsed
            return elapsed
        return 0.0

    def get_elapsed(self, name: str) -> float:
        return self.times.get(name, 0.0)

    def summary(self) -> str:
        lines = ["Timer Summary:", "-" * 30]
        for name, t in self.times.items():
            lines.append(f"  {name}: {t:.2f}s")
        return "\n".join(lines)

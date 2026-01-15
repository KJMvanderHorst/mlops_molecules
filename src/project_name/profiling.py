"""Profiling utilities for training and evaluation."""

from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

import time

import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler


class TrainingProfiler:
    """Manages profiling across entire training session."""

    def __init__(self, enabled: bool = False, output_dir: Optional[Path] = None):
        """Initialize the training profiler.

        Args:
            enabled: Whether to enable profiling.
            output_dir: Directory to save profiling results.
        """
        self.enabled = enabled
        self.output_dir = output_dir or Path("profiling_results/run")
        self.prof: Optional[profile] = None

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.prof = profile(
                activities=[ProfilerActivity.CPU]
                if not torch.cuda.is_available()
                else [ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                schedule=torch.profiler.schedule(
                    wait=0,
                    warmup=1,
                    active=10,
                    repeat=1,
                ),
                on_trace_ready=tensorboard_trace_handler(output_dir),
            )
            self.prof.__enter__()

    def step(self) -> None:
        """Record a step (epoch) in the profiler."""
        if self.prof:
            self.prof.step()

    def finalize(self) -> None:
        """Finalize profiling and export trace."""
        if self.prof:
            self.prof.__exit__(None, None, None)
            print(f"✅ Profiling trace saved to {self.output_dir}")


@contextmanager
def timing_checkpoint(name: str, enabled: bool = True) -> Generator:
    """Context manager for simple timing measurements.

    Args:
        name: Name for this checkpoint.
        enabled: Whether to enable timing.

    Yields:
        Dictionary with timing results.
    """
    result = {"name": name, "duration": 0.0}

    if not enabled:
        yield result
        return

    start = time.perf_counter()
    try:
        yield result
    finally:
        result["duration"] = time.perf_counter() - start
        print(f"⏱️  {name}: {result['duration']:.4f}s")

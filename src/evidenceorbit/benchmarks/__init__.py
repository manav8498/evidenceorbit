"""Public failure-mode benchmark for EvidenceOrbit v0.1."""

from evidenceorbit.benchmarks.corpus import CASES, BenchCase
from evidenceorbit.benchmarks.runner import BenchResult, run_benchmark

__all__ = ["CASES", "BenchCase", "BenchResult", "run_benchmark"]

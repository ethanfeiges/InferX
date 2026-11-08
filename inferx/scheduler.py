"""Decides when deferred memory work may run.

Version 1 has no device checks: the benchmark harness calls it at simulated idle
points. A real device policy would also check idle, charging and thermal state.
"""
from __future__ import annotations

from typing import Callable


class Scheduler:
    def __init__(self, may_run: Callable[[], bool] = lambda: True):
        self.may_run = may_run

    def run(self, work: Callable[[], dict]) -> dict:
        return work() if self.may_run() else {"skipped": True}

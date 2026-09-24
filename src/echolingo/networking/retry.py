from __future__ import annotations

import random
import time
from dataclasses import dataclass


@dataclass(slots=True)
class RetryPolicy:
    initial_s: float = 0.25
    maximum_s: float = 4.0
    budget_s: float = 30.0

    def delays(self, random_value=random.random, clock=time.monotonic):
        # The budget is wall time: the attempts between delays count, and so
        # does a coarse sleep timer (about 15 ms on Windows).
        started = clock()
        elapsed = 0.0
        ceiling = self.initial_s
        while elapsed < self.budget_s:
            delay = min(self.maximum_s, ceiling) * random_value()
            if elapsed + delay > self.budget_s:
                break
            yield delay
            elapsed = max(elapsed + delay, clock() - started)
            ceiling = min(self.maximum_s, ceiling * 2.0)

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(slots=True)
class RetryPolicy:
    initial_s: float = 0.25
    maximum_s: float = 4.0
    budget_s: float = 30.0

    def delays(self, random_value=random.random):
        elapsed = 0.0
        ceiling = self.initial_s
        while elapsed < self.budget_s:
            delay = min(self.maximum_s, ceiling) * random_value()
            if elapsed + delay > self.budget_s:
                break
            yield delay
            elapsed += delay
            ceiling = min(self.maximum_s, ceiling * 2.0)


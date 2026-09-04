"""
A class to implement a Cosine Decay learning rate
with linear warmup

Author: Mohanad Albughdadi
Created: 2025-09-12
"""

import math


class WarmupCosineLR:
    """
    Cosine decay learning rate with linear warmup.
    Step-based scheduler with optional full schedule generation.
    """

    def __init__(self, optimizer, total_steps, base_lr, warmup_steps=0, min_lr=0.0):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.last_step = 0

    def step(self, step=None):
        if step is None:
            step = self.last_step
        self.last_step = step

        # Linear warmup
        if step < self.warmup_steps and self.warmup_steps > 0:
            lr = self.base_lr * step / self.warmup_steps
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def get_full_schedule(self):
        """
        Returns a list of LR values for each step (0..total_steps-1).
        Useful for plotting or logging.
        """
        schedule = []
        for step in range(self.total_steps):
            if step < self.warmup_steps and self.warmup_steps > 0:
                lr = self.base_lr * step / self.warmup_steps
            else:
                progress = (step - self.warmup_steps) / max(
                    1, self.total_steps - self.warmup_steps
                )
                lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                    1 + math.cos(math.pi * progress)
                )
            schedule.append(lr)
        return schedule

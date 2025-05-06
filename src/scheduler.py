import math

import torch
import torch.optim.lr_scheduler


class CosineAnnealingWarmupLR(torch.optim.lr_scheduler.CosineAnnealingLR):

    def __init__(self, *args, warmup_steps: int = 2000, **kwargs):
        super().__init__(*args, **kwargs)
        self._warmup_steps = warmup_steps

    def get_last_lr(self):
        lr = super().get_last_lr()
        fac = min(1., self.last_epoch / self._warmup_steps)
        fac = math.sin(fac * math.pi / 2)
        return [i * fac for i in lr]

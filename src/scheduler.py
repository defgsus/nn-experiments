import math

import torch
import torch.optim.lr_scheduler


class CosineAnnealingWarmupLR(torch.optim.lr_scheduler.CosineAnnealingLR):

    def get_last_lr(self):
        lr = super().get_last_lr()
        fac = min(1., self.last_epoch / 2000)
        fac = math.sin(fac * math.pi / 2)
        return [i * fac for i in lr]

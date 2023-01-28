from torch.optim.lr_scheduler import _LRScheduler


class NoamLR(_LRScheduler):
    """Attention is all you need
    """
    def __init__(self, optimizer, d_model, warmup_steps) -> None:
        self.d_model = float(d_model)
        self.warmup_steps = float(warmup_steps)
        super().__init__(optimizer)
        
    def get_lr(self):
        curr_step = self.last_epoch + 1
        factor = min(curr_step ** (-0.5), curr_step * self.warmup_steps ** (-1.5))
        lrs = []
        for base_lr in self.base_lrs:
            lr = base_lr * self.d_model ** (-0.5) * factor
            lrs.append(lr)
        return lrs


class NoamDecayLR(_LRScheduler):
    """StyleGestures
    """
    def __init__(self, optimizer, warmup_steps, minimum=None):
        self.warmup_steps = float(warmup_steps)
        self.minimum = minimum
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        lrs = []
        for base_lr in self.base_lrs:
            lr = base_lr * scale
            if self.minimum is not None and last_epoch > self.warmup_steps:
                if lr < self.minimum:
                    lr = self.minimum
            lrs.append(lr)
        return lrs


class ConstantLR(_LRScheduler):
    def __init__(self, optimizer):
        super().__init__(optimizer)

    def get_lr(self):
        return self.base_lrs

    def load_state_dict(self, *args, **kwargs):
        print('[Warning] ConstantLR does not restore from state_dict.')
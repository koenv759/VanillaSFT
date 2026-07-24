"""
Minimal plugin for SFT: only registers FixLRSchedulerCallback.
No vLLM/GRPO imports — safe to load from swift sft.
"""

from swift.callbacks import callbacks_map
from swift.callbacks.base import TrainerCallback


class FixLRSchedulerCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        trainer = self.trainer
        scheduler = trainer.lr_scheduler
        optimizer = trainer.optimizer
        if scheduler is None or optimizer is None:
            return
        n_groups = len(optimizer.param_groups)
        # Check each attribute independently — on resume, lr_lambdas is re-created
        # at full length (not saved in checkpoint) while base_lrs comes from the
        # checkpoint already truncated, so the mismatch direction can differ.
        for attr in ('base_lrs', 'lr_lambdas', '_last_lr'):
            val = getattr(scheduler, attr, None)
            if val is not None and len(val) != n_groups:
                print(f'[fix_lr] {attr}: {len(val)} -> {n_groups}')
                setattr(scheduler, attr, val[:n_groups])


callbacks_map['fix_lr'] = FixLRSchedulerCallback

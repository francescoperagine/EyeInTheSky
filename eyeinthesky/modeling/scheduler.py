from torch.optim.lr_scheduler import LRScheduler
from ultralytics.utils import LOGGER
import math
import numpy as np

class DecayingCyclicalLRSchedulerCallback:
    """
    Callback that sets up the decaying cyclical learning rate scheduler.
    
    This callback integrates a custom learning rate scheduler into the training process
    by attaching it to the trainer during the pretrain routine. It forwards configuration
    parameters from the distillation config to the scheduler itself.
    
    Args:
        **kwargs: Configuration parameters from config["distillation"], including:
            max_lr (float): Maximum learning rate peak
            cycle_size (int): Number of epochs per cycle
            group_scalers (dict): Per-parameter group learning rate multipliers
    """
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def set_scheduler(self, trainer):
        trainer.scheduler = DecayingCyclicalLR(
            trainer=trainer, 
            **self.kwargs
        )

class DecayingCyclicalLR(LRScheduler):
    """
    Learning rate scheduler that combines cyclical patterns with gradual decay.
    
    Creates a wave-like learning rate pattern that oscillates between min_lr and max_lr
    while gradually decreasing the overall magnitude over time. Features include:
    
    1. Initial warmup phase with linear lr increase
    2. Cyclical pattern using cosine interpolation
    3. Gradual decay of cycle amplitude over training
    4. Support for different lr multipliers per parameter group
    
    Args:
        trainer: The trainer instance to access optimizer and training parameters
        **kwargs: Configuration parameters from config["distillation"], including:
            max_lr (float): Peak learning rate multiplier
            cycle_size (int): Number of epochs in each lr cycle
            group_scalers (dict): Parameter group-specific lr multipliers
    
    The scheduler integrates the trainer's built-in parameters like epochs, lrf (final lr factor),
    and warmup_epochs with the distillation-specific configuration.
    """

    def __init__(self, 
                 trainer, 
                 **kwargs
            ):
        self.optimizer = trainer.optimizer
        self.min_lr = trainer.args.lr0
        self.max_lr = trainer.args.lr0 * 10 if kwargs.get("max_lr") is None else kwargs.get("max_lr")
        self.lrf = trainer.args.lrf
        self.cycle_size = kwargs.get("cycle_size", 20)
        self.warmup_epochs = trainer.args.warmup_epochs if hasattr(trainer.args, 'warmup_epochs') else 0
        self.warmup_start_lr = trainer.args.warmup_start_lr if hasattr(trainer.args, 'warmup_start_lr') else self.min_lr * 0.1
        self.epochs = trainer.args.epochs

        self.group_scalers = kwargs.get("group_scalers") if kwargs.get("group_scalers") is not None else {0: 1.0, 1: 1.0, 2: 1.0}

        # Debug logging
        LOGGER.info(f"DecayingCyclicalLR: Min lr: {self.min_lr}, Max lr: {self.max_lr}, Cycle size: {self.cycle_size}, cycle size: {self.cycle_size}, lrf: {self.lrf}, group_scalers: {self.group_scalers}, warmup_epochs: {self.warmup_epochs}, warmup_start_lr: {self.warmup_start_lr}, epochs: {self.epochs}")

        super().__init__(self.optimizer)

    def get_lr(self):
        # During warmup: gradually increase from warmup_start_lr to min_lr over warmup_steps
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = self.last_epoch / max(1, self.warmup_epochs)
            base_lr = self.warmup_start_lr + (self.min_lr - self.warmup_start_lr) * warmup_factor
            LOGGER.info(f"DecayingCyclicalLR: Warmup epoch progression {self.last_epoch}/{self.warmup_epochs}")
            return [base_lr * self.group_scalers.get(i, 1.0) for i in range(len(self.optimizer.param_groups))]

        # After warmup: adjust using cyclical pattern with decay
        post_warmup_epoch = self.last_epoch - self.warmup_epochs
        remaining_epochs = self.epochs - self.warmup_epochs

        # Across-cycles decay (outer schedule)
        progress = post_warmup_epoch / max(1, remaining_epochs)
        progress = min(1.0, progress)  # cap at 1.0

        # decay_factor = self.lr_final_factor ** progress # Exponential decay
        # decay_factor = (1 - progress) ** 2 + self.lr_final_factor * progress # Quadratic decay
        # decay_factor = self.lr_final_factor + (1 - self.lr_final_factor) * 0.5 * (1 + math.cos(math.pi * progress)) # Cosine decay
        decay_factor = 1.0 - (1.0 - self.lrf) * progress  # Linear decay

        current_min_lr = self.min_lr * decay_factor
        current_max_lr = self.max_lr * decay_factor

        cycle_position = post_warmup_epoch % self.cycle_size

        # Cosine interpolation - adjusted to ensure proper cycle behavior
        # Produces a value that starts at 0, peaks at 1 at half cycle, and returns to 0
        normalized = cycle_position / self.cycle_size  # 0 to 1 over the cycle
        # Modified cosine formula: 0.5 * (1 - cos(2π * normalized))
        # At normalized=0: 0.5 * (1 - cos(0)) = 0.5 * (1 - 1) = 0
        # At normalized=0.5: 0.5 * (1 - cos(π)) = 0.5 * (1 - (-1)) = 1
        # At normalized=1: 0.5 * (1 - cos(2π)) = 0.5 * (1 - 1) = 0
        factor = 0.5 * (1 - math.cos(2 * math.pi * normalized))
        
        # Calculate learning rate
        base_lr = current_min_lr + (current_max_lr - current_min_lr) * factor

        LOGGER.info(f"\tPost-warmup epoch {post_warmup_epoch}, cycle pos {cycle_position}/{self.cycle_size}, decay_factor={decay_factor:.6f}")
        LOGGER.info(f"\tNormalized={normalized:.4f}, Factor={factor:.4f} min_lr={current_min_lr:.6f}, max_lr={current_max_lr:.6f}, base_lr={base_lr:.6f}")

        # Apply group-specific scaling
        groups_lrs = [np.float64(base_lr * self.group_scalers.get(i, 1.0)) for i in range(len(self.optimizer.param_groups))]
        return groups_lrs
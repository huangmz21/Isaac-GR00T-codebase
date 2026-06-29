"""
Updated DualBrainTrainer with V2 (sample-level queue) per-task loss tracking.
Drop-in replacement for the original trainer.
"""

import os
import torch
import transformers
from torch.utils.data import Dataset, Sampler
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    TRAINER_STATE_NAME,
    TrainerState,
    get_last_checkpoint,
    get_parameter_names,
    is_sagemaker_mp_enabled,
)

from gr00t.experiment.per_task_loss_tracker_v2 import PerTaskLossTrackerV2


class BaseSampler(Sampler):
    """Sampler for dataset, which enables `set_epoch` for Dataset."""

    def __init__(self, data_source: Dataset, shuffle: bool = False, seed: int = 0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            return iter(torch.randperm(len(self.data_source), generator=g).tolist())
        return iter(range(len(self.data_source)))

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.data_source, "set_epoch"):
            self.data_source.set_epoch(epoch)

    def __len__(self):
        return len(self.data_source)


class DualBrainTrainerWithPerTaskLoss(transformers.Trainer):
    """
    Enhanced DualBrainTrainer with V2 sample-level per-task loss tracking.

    Features:
    - Tracks EACH SAMPLE's loss individually in a FIFO queue
    - More precise: equal weight per sample (not per step)
    - More real-time: updates after every batch
    - Automatic trend detection: improving/stable/degrading
    - Supports both mixture and single datasets
    """

    def __init__(
        self,
        enable_per_task_tracking: bool = True,
        loss_queue_size: int = 1000,
        loss_log_interval: int = 10,
        min_samples_for_stats: int = 50,
        compute_per_sample_loss: bool = True,
        **kwargs
    ):
        """
        Args:
            enable_per_task_tracking: Enable per-task loss tracking
            loss_queue_size: Number of samples to keep per task (FIFO queue)
            loss_log_interval: Log per-task stats every N steps
            min_samples_for_stats: Minimum samples required before computing stats
            compute_per_sample_loss: Compute per-sample losses (required for tracking)
            **kwargs: Arguments for transformers.Trainer
        """
        self.compute_dtype = kwargs.pop("compute_dtype")
        self.enable_per_task_tracking = enable_per_task_tracking
        self.compute_per_sample_loss = compute_per_sample_loss

        super().__init__(**kwargs)

        # Initialize V2 tracker (sample-level queue)
        if self.enable_per_task_tracking:
            self.per_task_tracker = PerTaskLossTrackerV2(
                queue_size=loss_queue_size,
                log_interval=loss_log_interval,
                num_tasks=18,  # Default for atomic_seen
                min_samples_for_stats=min_samples_for_stats,
            )
            print(f"✅ Per-task loss tracking enabled (V2 - Sample-Level Queue)")
            print(f"   Queue size: {loss_queue_size} samples per task")
            print(f"   Log interval: every {loss_log_interval} steps")
        else:
            self.per_task_tracker = None

    def _get_train_sampler(self):
        return BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return BaseSampler(eval_dataset, shuffle=False)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with optional per-task tracking (V2).
        """
        # Extract dataset metadata before forward pass
        dataset_indices = inputs.pop("dataset_index", None) if self.enable_per_task_tracking else None
        dataset_names = inputs.pop("dataset_name", None) if self.enable_per_task_tracking else None

        # Forward pass
        outputs = model(inputs)
        loss = outputs["loss"]

        # Per-task loss tracking (V2 - sample-level)
        if self.enable_per_task_tracking and dataset_indices is not None:
            # Get per-sample losses
            if self.compute_per_sample_loss and "per_sample_loss" in outputs:
                per_sample_losses = outputs["per_sample_loss"]
            elif hasattr(outputs, "logits"):
                # Fallback: recompute per-sample loss if not provided by model
                per_sample_losses = self._compute_per_sample_loss(outputs, inputs)
            else:
                # If we can't get per-sample loss, use the mean loss for all samples
                # This is less accurate but still gives us task coverage info
                batch_size = dataset_indices.shape[0]
                per_sample_losses = loss.detach().expand(batch_size)

            # Update tracker - each sample goes into queue individually
            self.per_task_tracker.update(
                loss=per_sample_losses,
                dataset_indices=dataset_indices,
                dataset_names=dataset_names if isinstance(dataset_names, list) else None,
            )

            # Log statistics
            if self.per_task_tracker.should_log() and self.state.global_step > 0:
                stats = self.per_task_tracker.get_statistics()

                # Log to wandb/tensorboard
                if self.args.report_to:
                    self.log(stats)

                # Print summary every 100 steps
                if self.state.global_step % 100 == 0:
                    print("\n" + self.per_task_tracker.get_summary_report())

        return (loss, outputs) if return_outputs else loss

    def _compute_per_sample_loss(self, outputs, inputs):
        """
        Fallback method to compute per-sample loss.
        Override this based on your model's loss function.
        """
        # This is a placeholder - you need to implement based on your model
        # For flow matching models, this would recompute the loss per sample
        # For now, return uniform loss
        batch_size = inputs.get("batch_size", 1)
        return outputs["loss"].detach().expand(batch_size)

    def create_optimizer(self):
        """Setup the optimizer (same as original)."""
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def save_model(self, output_dir, _internal_call: bool):
        """Save model (same as original)."""
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            state_dict = self.model.state_dict()

        if self.args.should_save:
            return self.model.save_pretrained(output_dir, state_dict=state_dict)

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        """Train with checkpoint resumption (same as original)."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)


# Backward compatibility alias
DualBrainTrainer = DualBrainTrainerWithPerTaskLoss

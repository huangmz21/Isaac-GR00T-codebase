"""
Per-task loss tracker with sample-level queue (Version 2).

Key difference from V1:
- V1: Accumulates per-step average loss (window_size = number of steps)
- V2: Accumulates per-sample loss (queue_size = number of samples)

Advantages of V2:
- More fine-grained: tracks every sample individually
- More real-time: updates after every batch
- More accurate: each sample has equal weight
- More fair: not biased by batch composition
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, Optional, Union

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PerTaskLossTrackerV2:
    """
    Track per-task losses with sample-level queue.

    Each sample's loss is stored individually in a FIFO queue.
    When the queue is full, oldest samples are automatically removed.

    Usage:
        tracker = PerTaskLossTrackerV2(queue_size=1000, log_interval=10)

        # In training loop
        for batch in dataloader:
            loss = model(batch)
            # Update with per-sample losses
            tracker.update(per_sample_losses, dataset_indices, dataset_names)

            # Get real-time statistics
            if tracker.should_log():
                stats = tracker.get_statistics()
                # Log to wandb/tensorboard
    """

    def __init__(
        self,
        queue_size: int = 1000,
        log_interval: int = 10,
        num_tasks: int = 18,
        min_samples_for_stats: int = 10,
    ):
        """
        Args:
            queue_size: Maximum number of samples to keep per task (FIFO queue)
            log_interval: How often to log statistics (in steps)
            num_tasks: Total number of tasks
            min_samples_for_stats: Minimum samples required before computing stats
        """
        self.queue_size = queue_size
        self.log_interval = log_interval
        self.num_tasks = num_tasks
        self.min_samples_for_stats = min_samples_for_stats

        # Per-task sample queues: {task_id: deque of sample losses}
        # deque with maxlen automatically removes oldest when full (FIFO)
        self.task_queues = defaultdict(lambda: deque(maxlen=queue_size))

        # Per-task moving average (cached for efficiency)
        self.task_moving_avg = {}

        # Global step counter
        self.step = 0

        # Task name mapping
        self.task_id_to_name = {}

        # Statistics for monitoring
        self.total_samples_processed = 0
        self.samples_per_task = defaultdict(int)

    def update(
        self,
        loss: Union["torch.Tensor", np.ndarray],
        dataset_indices: Union["torch.Tensor", np.ndarray],
        dataset_names: Optional[list] = None,
    ):
        """
        Update tracker with current batch.

        Args:
            loss: Per-sample losses, shape (batch_size,)
            dataset_indices: Dataset index for each sample, shape (batch_size,)
            dataset_names: Optional list of dataset names for logging
        """
        self.step += 1

        # Convert to numpy for easier processing
        if TORCH_AVAILABLE and isinstance(loss, torch.Tensor):
            if loss.dim() == 0:  # scalar loss
                raise ValueError("Please pass per-sample losses, not scalar")
            loss = loss.detach().cpu().numpy()
        elif not isinstance(loss, np.ndarray):
            loss = np.array(loss)

        if TORCH_AVAILABLE and isinstance(dataset_indices, torch.Tensor):
            dataset_indices = dataset_indices.cpu().numpy()
        elif not isinstance(dataset_indices, np.ndarray):
            dataset_indices = np.array(dataset_indices)

        # Add each sample individually to its task queue
        for i in range(len(loss)):
            task_id = int(dataset_indices[i])
            sample_loss = float(loss[i])

            # Add to queue (automatically removes oldest if full)
            self.task_queues[task_id].append(sample_loss)

            # Update statistics
            self.total_samples_processed += 1
            self.samples_per_task[task_id] += 1

            # Update task name mapping
            if dataset_names and task_id not in self.task_id_to_name:
                if isinstance(dataset_names, list):
                    if len(dataset_names) == len(dataset_indices):
                        # Parallel to dataset_indices
                        self.task_id_to_name[task_id] = dataset_names[i]
                    elif len(dataset_names) > task_id:
                        # Indexed by task_id
                        self.task_id_to_name[task_id] = dataset_names[task_id]

    def get_statistics(self, return_dict: bool = True) -> Dict[str, float]:
        """
        Compute real-time statistics for all tasks.

        Returns:
            Dictionary with per-task statistics:
            - task_{name}/loss_avg: average loss in queue
            - task_{name}/loss_std: standard deviation in queue
            - task_{name}/loss_min: minimum loss in queue
            - task_{name}/loss_max: maximum loss in queue
            - task_{name}/queue_size: current number of samples in queue
            - task_{name}/total_samples: total samples processed (lifetime)
            - task_{name}/queue_utilization: queue_size / max_queue_size
        """
        stats = {}

        for task_id, queue in self.task_queues.items():
            if len(queue) < self.min_samples_for_stats:
                continue

            # Convert queue to numpy array
            losses = np.array(list(queue))

            # Compute statistics
            avg_loss = np.mean(losses)
            std_loss = np.std(losses)
            min_loss = np.min(losses)
            max_loss = np.max(losses)
            queue_size = len(losses)
            total_samples = self.samples_per_task[task_id]
            utilization = queue_size / self.queue_size

            # Get task name
            task_name = self.task_id_to_name.get(task_id, f"task_{task_id}")

            stats[f"{task_name}/loss_avg"] = float(avg_loss)
            stats[f"{task_name}/loss_std"] = float(std_loss)
            stats[f"{task_name}/loss_min"] = float(min_loss)
            stats[f"{task_name}/loss_max"] = float(max_loss)
            stats[f"{task_name}/queue_size"] = int(queue_size)
            stats[f"{task_name}/total_samples"] = int(total_samples)
            stats[f"{task_name}/queue_utilization"] = float(utilization)

            # Update moving average cache
            self.task_moving_avg[task_id] = avg_loss

        # Add global statistics
        if len(self.task_moving_avg) > 0:
            all_avgs = list(self.task_moving_avg.values())
            stats["global/task_loss_mean"] = float(np.mean(all_avgs))
            stats["global/task_loss_std"] = float(np.std(all_avgs))
            stats["global/task_loss_min"] = float(np.min(all_avgs))
            stats["global/task_loss_max"] = float(np.max(all_avgs))
            stats["global/num_active_tasks"] = len(all_avgs)
            stats["global/total_samples"] = int(self.total_samples_processed)

            # Queue statistics
            queue_sizes = [len(q) for q in self.task_queues.values()]
            stats["global/avg_queue_size"] = float(np.mean(queue_sizes))
            stats["global/min_queue_size"] = int(np.min(queue_sizes))
            stats["global/max_queue_size"] = int(np.max(queue_sizes))

        return stats

    def get_task_ranks(self) -> Dict[str, int]:
        """
        Get task ranking by loss (1 = lowest loss, best performing).
        """
        if not self.task_moving_avg:
            return {}

        # Sort tasks by loss
        sorted_tasks = sorted(self.task_moving_avg.items(), key=lambda x: x[1])

        ranks = {}
        for rank, (task_id, _) in enumerate(sorted_tasks, start=1):
            task_name = self.task_id_to_name.get(task_id, f"task_{task_id}")
            ranks[task_name] = rank

        return ranks

    def get_task_trends(self, window: int = 100) -> Dict[str, str]:
        """
        Analyze if each task's loss is improving, stable, or degrading.

        Args:
            window: Number of recent samples to compare with older samples

        Returns:
            Dictionary mapping task_name to trend: "improving", "stable", "degrading"
        """
        trends = {}

        for task_id, queue in self.task_queues.items():
            if len(queue) < window * 2:
                continue  # Need at least 2x window size

            losses = np.array(list(queue))
            # Compare first half vs second half
            older_half = losses[:len(losses)//2]
            recent_half = losses[len(losses)//2:]

            older_avg = np.mean(older_half)
            recent_avg = np.mean(recent_half)

            # Compute relative change
            rel_change = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0

            task_name = self.task_id_to_name.get(task_id, f"task_{task_id}")

            if rel_change < -0.05:  # Improved by > 5%
                trends[task_name] = "improving ↓"
            elif rel_change > 0.05:  # Degraded by > 5%
                trends[task_name] = "degrading ↑"
            else:
                trends[task_name] = "stable →"

        return trends

    def should_log(self) -> bool:
        """Check if we should log at this step."""
        return self.step % self.log_interval == 0

    def reset(self):
        """Reset all statistics."""
        self.task_queues.clear()
        self.task_moving_avg.clear()
        self.task_id_to_name.clear()
        self.samples_per_task.clear()
        self.total_samples_processed = 0
        self.step = 0

    def get_summary_report(self) -> str:
        """
        Generate a human-readable summary report.
        """
        stats = self.get_statistics()
        ranks = self.get_task_ranks()
        trends = self.get_task_trends()

        lines = []
        lines.append("=" * 80)
        lines.append(f"📊 Per-Task Loss Summary (Step {self.step})")
        lines.append("=" * 80)

        # Extract task losses
        task_losses = {}
        for key, value in stats.items():
            if key.endswith("/loss_avg"):
                task_name = key.replace("/loss_avg", "")
                task_losses[task_name] = value

        if not task_losses:
            lines.append("⚠️  No task data available yet")
            return "\n".join(lines)

        # Sort by loss
        sorted_tasks = sorted(task_losses.items(), key=lambda x: x[1])

        # Best performing
        lines.append(f"\n🎯 Best performing tasks (lowest loss):")
        for task_name, loss in sorted_tasks[:3]:
            queue_size = stats.get(f"{task_name}/queue_size", 0)
            trend = trends.get(task_name, "")
            lines.append(f"  ✅ {task_name}: {loss:.4f} ({queue_size} samples) {trend}")

        # Worst performing
        lines.append(f"\n⚠️  Worst performing tasks (highest loss):")
        for task_name, loss in sorted_tasks[-3:]:
            queue_size = stats.get(f"{task_name}/queue_size", 0)
            trend = trends.get(task_name, "")
            lines.append(f"  ❌ {task_name}: {loss:.4f} ({queue_size} samples) {trend}")

        # Global stats
        lines.append(f"\n📈 Global Statistics:")
        lines.append(f"  Mean loss across tasks: {stats.get('global/task_loss_mean', 0):.4f}")
        lines.append(f"  Std loss across tasks: {stats.get('global/task_loss_std', 0):.4f}")
        lines.append(f"  Active tasks: {stats.get('global/num_active_tasks', 0)}/{self.num_tasks}")
        lines.append(f"  Total samples processed: {stats.get('global/total_samples', 0)}")
        lines.append(f"  Avg queue utilization: {stats.get('global/avg_queue_size', 0) / self.queue_size * 100:.1f}%")

        lines.append("=" * 80)

        return "\n".join(lines)


# Backward compatibility
PerTaskLossTracker = PerTaskLossTrackerV2

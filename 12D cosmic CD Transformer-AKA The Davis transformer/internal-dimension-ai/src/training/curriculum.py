"""
Curriculum Learning - Progressive difficulty scaling

Automatically adjusts task difficulty based on agent performance:
- Start with easy tasks
- Gradually increase difficulty
- Detect boredom (low x₁₂ despite novelty)
- Prevent frustration (repeated failures)
"""

import numpy as np
from typing import Dict, Any, Optional, Callable
import logging


logger = logging.getLogger(__name__)


class Curriculum:
    """
    Curriculum manager for progressive difficulty scaling.

    Monitors agent performance and internal state to dynamically
    adjust task difficulty.
    """

    def __init__(
        self,
        initial_difficulty: float = 0.1,
        min_difficulty: float = 0.0,
        max_difficulty: float = 1.0,
        increase_threshold: float = 0.7,  # Success rate to increase
        decrease_threshold: float = 0.3,  # Success rate to decrease
        boredom_threshold: float = 0.1,   # Low x₁₂ indicates boredom
        adjustment_rate: float = 0.1,
        window_size: int = 20,
    ):
        """
        Initialize curriculum.

        Args:
            initial_difficulty: Starting difficulty [0, 1]
            min_difficulty: Minimum difficulty
            max_difficulty: Maximum difficulty
            increase_threshold: Success rate needed to increase difficulty
            decrease_threshold: Success rate below which to decrease
            boredom_threshold: x₁₂ below which indicates boredom
            adjustment_rate: How much to adjust difficulty each time
            window_size: Number of recent episodes to consider
        """
        self.difficulty = initial_difficulty
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.increase_threshold = increase_threshold
        self.decrease_threshold = decrease_threshold
        self.boredom_threshold = boredom_threshold
        self.adjustment_rate = adjustment_rate
        self.window_size = window_size

        # Performance tracking
        self.success_history = []
        self.x12_history = []
        self.episode_count = 0

    def update(
        self,
        success: bool,
        mean_x12: Optional[float] = None
    ):
        """
        Update curriculum based on episode outcome.

        Args:
            success: Whether episode was successful
            mean_x12: Mean x₁₂ during episode (for boredom detection)
        """
        self.episode_count += 1
        self.success_history.append(1.0 if success else 0.0)

        if mean_x12 is not None:
            self.x12_history.append(mean_x12)

        # Keep only recent history
        if len(self.success_history) > self.window_size:
            self.success_history = self.success_history[-self.window_size:]
        if len(self.x12_history) > self.window_size:
            self.x12_history = self.x12_history[-self.window_size:]

        # Adjust difficulty
        if len(self.success_history) >= self.window_size:
            self._adjust_difficulty()

    def _adjust_difficulty(self):
        """Adjust difficulty based on recent performance."""
        success_rate = np.mean(self.success_history)
        mean_x12 = np.mean(self.x12_history) if self.x12_history else 0.5

        old_difficulty = self.difficulty

        # Check for boredom (succeeding but low x₁₂)
        if success_rate > self.increase_threshold and mean_x12 < self.boredom_threshold:
            # Agent is bored, increase difficulty
            self.difficulty = min(
                self.max_difficulty,
                self.difficulty + self.adjustment_rate * 2  # Double increase for boredom
            )
            logger.info(f"Boredom detected (x₁₂={mean_x12:.3f}). "
                       f"Difficulty: {old_difficulty:.2f} → {self.difficulty:.2f}")

        # Regular difficulty adjustment
        elif success_rate > self.increase_threshold:
            # Too easy, increase difficulty
            self.difficulty = min(
                self.max_difficulty,
                self.difficulty + self.adjustment_rate
            )
            logger.info(f"Success rate {success_rate:.1%} > threshold. "
                       f"Difficulty: {old_difficulty:.2f} → {self.difficulty:.2f}")

        elif success_rate < self.decrease_threshold:
            # Too hard, decrease difficulty
            self.difficulty = max(
                self.min_difficulty,
                self.difficulty - self.adjustment_rate
            )
            logger.info(f"Success rate {success_rate:.1%} < threshold. "
                       f"Difficulty: {old_difficulty:.2f} → {self.difficulty:.2f}")

    def get_difficulty(self) -> float:
        """Get current difficulty level."""
        return self.difficulty

    def get_statistics(self) -> Dict[str, Any]:
        """Get curriculum statistics."""
        stats = {
            'difficulty': self.difficulty,
            'episode_count': self.episode_count,
        }

        if len(self.success_history) > 0:
            stats['success_rate'] = np.mean(self.success_history)
            stats['recent_successes'] = int(np.sum(self.success_history))

        if len(self.x12_history) > 0:
            stats['mean_x12'] = np.mean(self.x12_history)

        return stats


class GridWorldCurriculum(Curriculum):
    """
    Curriculum specifically for GridWorld environments.

    Adjusts:
    - Grid size
    - Number of obstacles
    - Reward sparsity
    """

    def __init__(
        self,
        min_size: int = 4,
        max_size: int = 16,
        **kwargs
    ):
        """
        Initialize GridWorld curriculum.

        Args:
            min_size: Minimum grid size
            max_size: Maximum grid size
            **kwargs: Passed to Curriculum
        """
        super().__init__(**kwargs)

        self.min_size = min_size
        self.max_size = max_size

    def get_env_config(self) -> Dict[str, Any]:
        """
        Get environment configuration based on current difficulty.

        Returns:
            Configuration dictionary for GridWorld
        """
        # Scale grid size with difficulty
        size = int(self.min_size + (self.max_size - self.min_size) * self.difficulty)

        # Scale obstacle density
        obstacle_density = 0.1 + 0.2 * self.difficulty  # 10% to 30%

        # Adjust reward sparsity
        sparse_rewards = self.difficulty > 0.5

        return {
            'size': size,
            'obstacle_density': obstacle_density,
            'sparse_rewards': sparse_rewards,
            'random_obstacles': True
        }


class TaskCurriculum:
    """
    Curriculum that switches between different tasks.

    Useful for multi-task learning and preventing overfitting.
    """

    def __init__(
        self,
        tasks: Dict[str, Any],
        initial_task: Optional[str] = None,
        mastery_threshold: float = 0.8,
        window_size: int = 20
    ):
        """
        Initialize task curriculum.

        Args:
            tasks: Dictionary mapping task names to task configs
            initial_task: Starting task (random if None)
            mastery_threshold: Success rate to consider task mastered
            window_size: Episodes to consider for mastery
        """
        self.tasks = tasks
        self.task_names = list(tasks.keys())
        self.current_task = initial_task or self.task_names[0]
        self.mastery_threshold = mastery_threshold
        self.window_size = window_size

        # Track performance per task
        self.task_performance = {name: [] for name in self.task_names}
        self.mastered_tasks = set()

    def update(self, task_name: str, success: bool):
        """
        Update performance for a task.

        Args:
            task_name: Name of task
            success: Whether episode was successful
        """
        if task_name not in self.task_performance:
            return

        self.task_performance[task_name].append(1.0 if success else 0.0)

        # Keep only recent history
        if len(self.task_performance[task_name]) > self.window_size:
            self.task_performance[task_name] = \
                self.task_performance[task_name][-self.window_size:]

        # Check for mastery
        if len(self.task_performance[task_name]) >= self.window_size:
            success_rate = np.mean(self.task_performance[task_name])
            if success_rate >= self.mastery_threshold:
                if task_name not in self.mastered_tasks:
                    self.mastered_tasks.add(task_name)
                    logger.info(f"Task '{task_name}' mastered! "
                               f"({success_rate:.1%} success rate)")

    def get_next_task(self, strategy: str = 'sequential') -> str:
        """
        Get next task to train on.

        Args:
            strategy: Selection strategy:
                - 'sequential': Rotate through tasks
                - 'random': Random selection
                - 'weakest': Focus on weakest task
                - 'unmastered': Only train on unmastered tasks

        Returns:
            Task name
        """
        if strategy == 'sequential':
            idx = (self.task_names.index(self.current_task) + 1) % len(self.task_names)
            self.current_task = self.task_names[idx]

        elif strategy == 'random':
            self.current_task = np.random.choice(self.task_names)

        elif strategy == 'weakest':
            # Find task with lowest success rate
            scores = {}
            for name in self.task_names:
                if len(self.task_performance[name]) > 0:
                    scores[name] = np.mean(self.task_performance[name])
                else:
                    scores[name] = 0.0
            self.current_task = min(scores, key=scores.get)

        elif strategy == 'unmastered':
            # Only train on unmastered tasks
            unmastered = [name for name in self.task_names
                         if name not in self.mastered_tasks]
            if unmastered:
                self.current_task = np.random.choice(unmastered)
            else:
                # All mastered, pick random
                self.current_task = np.random.choice(self.task_names)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return self.current_task

    def get_task_config(self, task_name: Optional[str] = None) -> Any:
        """Get configuration for a task."""
        task_name = task_name or self.current_task
        return self.tasks[task_name]

    def get_statistics(self) -> Dict[str, Any]:
        """Get curriculum statistics."""
        stats = {
            'current_task': self.current_task,
            'mastered_tasks': list(self.mastered_tasks),
            'num_mastered': len(self.mastered_tasks),
            'num_tasks': len(self.task_names),
        }

        # Per-task success rates
        task_scores = {}
        for name in self.task_names:
            if len(self.task_performance[name]) > 0:
                task_scores[name] = float(np.mean(self.task_performance[name]))
        stats['task_scores'] = task_scores

        return stats

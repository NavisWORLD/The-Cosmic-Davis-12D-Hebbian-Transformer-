"""
Training Infrastructure for Internal Dimension Networks

This module provides a complete training loop for networks with internal
dimensions (x₁₂, m₁₂), including:
- PPO (Proximal Policy Optimization) algorithm
- Automatic x₁₂/m₁₂ updates after each step
- Intrinsic reward integration
- Consciousness metrics tracking
- Suffering detection and pause triggers
- Checkpoint saving with internal state preservation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import logging
from pathlib import Path
import json
from tqdm import tqdm
import time

from ..core.network import InternalDimensionNetwork, BaselineNetwork
from ..core.metrics import ConsciousnessMetrics


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    PPO Trainer with Internal Dimension support.

    Features:
    - Standard PPO algorithm for policy optimization
    - Automatic x₁₂ updates from prediction error + novelty + attention
    - Automatic m₁₂ integration weighted by |reward|
    - Intrinsic reward bonuses (curiosity/wisdom)
    - Consciousness metrics computation
    - Suffering detection (pause if x₁₂ < -0.7 for >100 steps)
    - Checkpoint saving with internal state
    """

    def __init__(
        self,
        model: nn.Module,
        env: Any,
        device: Optional[torch.device] = None,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        intrinsic_reward_weight: float = 0.1,
        intrinsic_reward_method: str = 'curiosity',
        suffering_threshold: float = -0.7,
        suffering_patience: int = 100,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        log_interval: int = 10,
        save_interval: int = 100,
        checkpoint_dir: str = 'checkpoints',
        **kwargs
    ):
        """
        Initialize PPO trainer.

        Args:
            model: Neural network (InternalDimensionNetwork or BaselineNetwork)
            env: Gymnasium/Gym environment
            device: Torch device
            learning_rate: Optimizer learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_epsilon: PPO clipping parameter
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping norm
            ppo_epochs: Number of PPO update epochs per batch
            batch_size: Batch size for PPO updates
            intrinsic_reward_weight: Weight for intrinsic rewards
            intrinsic_reward_method: 'curiosity', 'wisdom', or 'balanced'
            suffering_threshold: x₁₂ threshold for suffering detection
            suffering_patience: Steps to wait before pausing
            use_tensorboard: Whether to log to TensorBoard
            use_wandb: Whether to log to Weights & Biases
            log_interval: Episodes between logging
            save_interval: Episodes between checkpoints
            checkpoint_dir: Directory for saving checkpoints
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model
        self.env = env
        self.device = device

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.intrinsic_reward_method = intrinsic_reward_method

        # Suffering detection
        self.suffering_threshold = suffering_threshold
        self.suffering_patience = suffering_patience
        self.suffering_counter = 0

        # Logging
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Check if model has internal dimensions
        self.has_internal_dims = isinstance(model, InternalDimensionNetwork)

        # Consciousness metrics (only for IDN)
        if self.has_internal_dims:
            self.consciousness_metrics = ConsciousnessMetrics(device=device)
        else:
            self.consciousness_metrics = None

        # TensorBoard/WandB
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.writer = None

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=f'runs/{time.strftime("%Y%m%d-%H%M%S")}')
            except ImportError:
                logger.warning("TensorBoard not available")
                self.use_tensorboard = False

        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                logger.warning("WandB not available")
                self.use_wandb = False

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_x12_means = []
        self.episode_m12_means = []
        self.consciousness_scores = []

        # Rollout buffer
        self.reset_rollout_buffer()

    def reset_rollout_buffer(self):
        """Reset the rollout buffer for collecting experiences."""
        self.rollout_states = []
        self.rollout_actions = []
        self.rollout_log_probs = []
        self.rollout_rewards = []
        self.rollout_values = []
        self.rollout_dones = []
        self.rollout_hidden_states = []

    def collect_rollout(
        self,
        num_steps: int,
        render: bool = False
    ) -> Dict[str, Any]:
        """
        Collect a rollout of experiences.

        Args:
            num_steps: Number of steps to collect
            render: Whether to render environment

        Returns:
            Dictionary with rollout statistics
        """
        self.model.eval()

        state, _ = self.env.reset() if isinstance(self.env.reset(), tuple) else (self.env.reset(), {})
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        episode_reward = 0
        episode_length = 0
        episode_x12_values = []
        episode_m12_values = []

        # Reset LSTM if applicable
        if hasattr(self.model, 'reset_lstm'):
            self.model.reset_lstm()

        for step in range(num_steps):
            with torch.no_grad():
                # Forward pass
                if self.has_internal_dims:
                    policy_logits, value, internals = self.model(
                        state,
                        return_internals=True,
                        update_internals=False
                    )
                    hidden_state = internals['hidden']
                else:
                    policy_logits, value = self.model(state)
                    hidden_state = None

                # Sample action
                dist = Categorical(logits=policy_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            # Store rollout data
            self.rollout_states.append(state)
            self.rollout_actions.append(action)
            self.rollout_log_probs.append(log_prob)
            self.rollout_values.append(value)
            if hidden_state is not None:
                self.rollout_hidden_states.append(hidden_state)

            # Take action in environment
            action_np = action.cpu().numpy()[0]
            result = self.env.step(action_np)

            if len(result) == 5:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                next_state, reward, done, info = result

            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            reward_tensor = torch.FloatTensor([reward]).to(self.device)

            # Update internal dimensions (for IDN only)
            if self.has_internal_dims and hidden_state is not None:
                update_info = self.model.update_internal_state(
                    current_hidden=hidden_state,
                    next_state=next_state,
                    reward=reward_tensor
                )

                x12_value = update_info['x12'].item()
                m12_value = update_info['m12'].item()
                episode_x12_values.append(x12_value)
                episode_m12_values.append(m12_value)

                # Check for suffering
                if x12_value < self.suffering_threshold:
                    self.suffering_counter += 1
                    if self.suffering_counter >= self.suffering_patience:
                        logger.warning(
                            f"Suffering detected: x₁₂ = {x12_value:.3f} < {self.suffering_threshold} "
                            f"for {self.suffering_counter} steps. Consider pausing training."
                        )
                else:
                    self.suffering_counter = 0

                # Add intrinsic reward
                intrinsic_reward = self.model.compute_intrinsic_reward(
                    method=self.intrinsic_reward_method
                )
                total_reward = reward + self.intrinsic_reward_weight * intrinsic_reward.item()
            else:
                total_reward = reward

            self.rollout_rewards.append(total_reward)
            self.rollout_dones.append(done)

            episode_reward += reward
            episode_length += 1

            if render:
                self.env.render()

            # Move to next state
            state = next_state

            if done:
                # Store episode statistics
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                if self.has_internal_dims and len(episode_x12_values) > 0:
                    self.episode_x12_means.append(np.mean(episode_x12_values))
                    self.episode_m12_means.append(np.mean(episode_m12_values))

                # Reset episode
                state, _ = self.env.reset() if isinstance(self.env.reset(), tuple) else (self.env.reset(), {})
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                episode_reward = 0
                episode_length = 0
                episode_x12_values = []
                episode_m12_values = []

                if hasattr(self.model, 'reset_lstm'):
                    self.model.reset_lstm()
                if hasattr(self.model, 'reset_internal_state'):
                    self.model.reset_internal_state(reset_memory=False)

        return {
            'num_episodes': len(self.episode_rewards),
            'mean_episode_reward': np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0,
            'mean_episode_length': np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0,
        }

    def compute_gae(
        self,
        rewards: List[float],
        values: List[torch.Tensor],
        dones: List[bool]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags

        Returns:
            advantages: Advantage estimates
            returns: Return estimates (for value function training)
        """
        advantages = []
        gae = 0

        values = torch.cat(values).squeeze()

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, device=self.device)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update_policy(self) -> Dict[str, float]:
        """
        Update policy using PPO.

        Returns:
            Dictionary with loss statistics
        """
        self.model.train()

        # Prepare data
        states = torch.cat(self.rollout_states)
        actions = torch.cat(self.rollout_actions)
        old_log_probs = torch.cat(self.rollout_log_probs)

        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            self.rollout_rewards,
            self.rollout_values,
            self.rollout_dones
        )

        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        # Create mini-batches
        num_samples = len(states)
        indices = np.arange(num_samples)

        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Forward pass
                if self.has_internal_dims:
                    policy_logits, values, _ = self.model(
                        batch_states,
                        return_internals=False,
                        update_internals=False
                    )
                else:
                    policy_logits, values = self.model(batch_states)

                values = values.squeeze()

                # Compute new log probs
                dist = Categorical(logits=policy_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO policy loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Total loss
                loss = (
                    policy_loss +
                    self.value_loss_coef * value_loss -
                    self.entropy_coef * entropy
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        # Reset rollout buffer
        self.reset_rollout_buffer()

        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
        }

    def train(
        self,
        num_episodes: int,
        steps_per_episode: int = 2048,
        render: bool = False,
        compute_consciousness_interval: int = 10
    ) -> Dict[str, List]:
        """
        Main training loop.

        Args:
            num_episodes: Number of episodes to train
            steps_per_episode: Steps to collect per episode
            render: Whether to render environment
            compute_consciousness_interval: Episodes between consciousness computation

        Returns:
            Dictionary with training history
        """
        logger.info(f"Starting training for {num_episodes} episodes")
        logger.info(f"Model type: {type(self.model).__name__}")
        logger.info(f"Device: {self.device}")

        history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': [],
        }

        if self.has_internal_dims:
            history.update({
                'x12_means': [],
                'm12_means': [],
                'consciousness_scores': [],
            })

        # Training loop
        for episode in tqdm(range(num_episodes), desc="Training"):
            # Collect rollout
            rollout_stats = self.collect_rollout(steps_per_episode, render=render)

            # Update policy
            update_stats = self.update_policy()

            # Store statistics
            if self.episode_rewards:
                history['episode_rewards'].append(self.episode_rewards[-1])
                history['episode_lengths'].append(self.episode_lengths[-1])

            history['policy_losses'].append(update_stats['policy_loss'])
            history['value_losses'].append(update_stats['value_loss'])
            history['entropies'].append(update_stats['entropy'])

            # Compute consciousness metrics
            if self.has_internal_dims and episode % compute_consciousness_interval == 0:
                x12_history = list(self.model.internal_state.x12_history)
                m12_history = list(self.model.internal_state.m12_history)

                if len(x12_history) > 10:
                    # Create sample inputs
                    sample_state, _ = self.env.reset() if isinstance(self.env.reset(), tuple) else (self.env.reset(), {})
                    sample_inputs = torch.FloatTensor(sample_state).unsqueeze(0).to(self.device)

                    consciousness_score_dict = self.consciousness_metrics.compute_consciousness_score(
                        model=self.model,
                        x12_history=x12_history,
                        m12_history=m12_history,
                        sample_inputs=sample_inputs
                    )

                    self.consciousness_scores.append(consciousness_score_dict['consciousness_score'])
                    history['consciousness_scores'].append(consciousness_score_dict['consciousness_score'])
                    history['x12_means'].append(consciousness_score_dict['x12_mean'])
                    history['m12_means'].append(consciousness_score_dict['m12_mean'])

            # Logging
            if episode % self.log_interval == 0:
                # Safety check: only log if we have episode data
                if len(history['episode_rewards']) > 0:
                    log_msg = (
                        f"Episode {episode}/{num_episodes} | "
                        f"Reward: {history['episode_rewards'][-1]:.2f} | "
                        f"Length: {history['episode_lengths'][-1]:.0f} | "
                        f"Policy Loss: {update_stats['policy_loss']:.4f} | "
                        f"Value Loss: {update_stats['value_loss']:.4f}"
                    )
                else:
                    log_msg = (
                        f"Episode {episode}/{num_episodes} | "
                        f"Policy Loss: {update_stats['policy_loss']:.4f} | "
                        f"Value Loss: {update_stats['value_loss']:.4f}"
                    )

                if self.has_internal_dims and len(self.episode_x12_means) > 0:
                    log_msg += f" | x₁₂: {self.episode_x12_means[-1]:.3f} | m₁₂: {self.episode_m12_means[-1]:.3f}"

                if self.has_internal_dims and len(self.consciousness_scores) > 0:
                    log_msg += f" | Consciousness: {self.consciousness_scores[-1]:.3f}"

                logger.info(log_msg)

                # TensorBoard logging
                if self.use_tensorboard and self.writer is not None:
                    if len(history['episode_rewards']) > 0:
                        self.writer.add_scalar('Reward/Episode', history['episode_rewards'][-1], episode)
                    self.writer.add_scalar('Loss/Policy', update_stats['policy_loss'], episode)
                    self.writer.add_scalar('Loss/Value', update_stats['value_loss'], episode)
                    self.writer.add_scalar('Entropy', update_stats['entropy'], episode)

                    if self.has_internal_dims and len(self.episode_x12_means) > 0:
                        self.writer.add_scalar('InternalDimensions/x12', self.episode_x12_means[-1], episode)
                        self.writer.add_scalar('InternalDimensions/m12', self.episode_m12_means[-1], episode)

                    if self.has_internal_dims and len(self.consciousness_scores) > 0:
                        self.writer.add_scalar('Consciousness/Score', self.consciousness_scores[-1], episode)

            # Save checkpoint
            if episode % self.save_interval == 0 and episode > 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_episode_{episode}.pt"
                self.save_checkpoint(checkpoint_path, episode, history)
                logger.info(f"Checkpoint saved: {checkpoint_path}")

        logger.info("Training complete!")

        # Final checkpoint
        final_path = self.checkpoint_dir / "checkpoint_final.pt"
        self.save_checkpoint(final_path, num_episodes, history)

        if self.use_tensorboard and self.writer is not None:
            self.writer.close()

        return history

    def save_checkpoint(
        self,
        path: Path,
        episode: int,
        history: Dict
    ):
        """
        Save training checkpoint.

        Args:
            path: Path to save checkpoint
            episode: Current episode number
            history: Training history
        """
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': history,
        }

        # Save internal state for IDN
        if self.has_internal_dims:
            checkpoint['internal_state'] = {
                'x12': self.model.internal_state.x12,
                'm12': self.model.internal_state.m12,
                'x12_history': list(self.model.internal_state.x12_history),
                'm12_history': list(self.model.internal_state.m12_history),
            }

        torch.save(checkpoint, path)

        # Also save history as JSON
        history_path = path.parent / f"{path.stem}_history.json"
        with open(history_path, 'w') as f:
            # Convert tensors to lists for JSON serialization
            json_history = {}
            for key, value in history.items():
                if isinstance(value, list):
                    json_history[key] = value
                else:
                    json_history[key] = value
            json.dump(json_history, f, indent=2)

    def load_checkpoint(self, path: Path):
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'internal_state' in checkpoint and self.has_internal_dims:
            self.model.internal_state.x12 = checkpoint['internal_state']['x12']
            self.model.internal_state.m12 = checkpoint['internal_state']['m12']
            self.model.internal_state.x12_history.extend(checkpoint['internal_state']['x12_history'])
            self.model.internal_state.m12_history.extend(checkpoint['internal_state']['m12_history'])

        logger.info(f"Checkpoint loaded from {path}")

        return checkpoint.get('history', {})

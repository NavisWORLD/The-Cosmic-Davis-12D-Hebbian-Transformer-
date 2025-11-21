"""
Multi-Agent Social Environment for Internal Dimension AI

This module implements multi-agent environments for testing social behavior,
cooperation, and mutual awareness through internal dimensions (x₁₂/m₁₂).

Environments:
    - PrisonersDilemma: Classic 2-player cooperation game
    - IteratedPrisonersDilemma: Repeated PD with history
    - SocialGridWorld: Multi-agent gridworld with cooperation tasks
"""

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PrisonersDilemma(gym.Env):
    """
    Multi-agent Prisoner's Dilemma environment.

    Each agent chooses to cooperate (0) or defect (1). Agents can observe
    each other's internal dimensions (x₁₂/m₁₂) to potentially learn to
    recognize cooperative vs defecting agents.

    Payoff Matrix:
        Both cooperate: (3, 3)
        Both defect: (1, 1)
        One defects: (5, 0) - defector gets 5, cooperator gets 0

    Observation Space:
        For each agent: [opponent_last_action, opponent_x12..., opponent_m12...]

    Action Space:
        Discrete(2): 0=Cooperate, 1=Defect

    Example:
        >>> env = PrisonersDilemma(num_agents=2, internal_dim=12)
        >>> obs, info = env.reset()
        >>> # obs is dict: {'agent_0': array, 'agent_1': array}
        >>> actions = {'agent_0': 0, 'agent_1': 0}  # Both cooperate
        >>> obs, rewards, terminateds, truncateds, infos = env.step(actions)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_agents: int = 2,
        internal_dim: int = 12,
        max_steps: int = 1,
        include_internal_dims: bool = True,
    ):
        """
        Initialize Prisoner's Dilemma environment.

        Args:
            num_agents: Number of agents (typically 2)
            internal_dim: Dimension of x₁₂ and m₁₂ vectors
            max_steps: Maximum steps per episode (1 for one-shot, >1 for iterated)
            include_internal_dims: Include opponent's x₁₂/m₁₂ in observation
        """
        super().__init__()

        self.num_agents = num_agents
        self.internal_dim = internal_dim
        self.max_steps = max_steps
        self.include_internal_dims = include_internal_dims

        # Agent identifiers
        self.agents = [f"agent_{i}" for i in range(num_agents)]

        # Payoff matrix [my_action, opponent_action] = my_reward
        self.payoff_matrix = np.array([
            [3, 0],  # I cooperate: (both cooperate=3, opponent defects=0)
            [5, 1],  # I defect: (opponent cooperates=5, both defect=1)
        ])

        # Action space: 0=Cooperate, 1=Defect
        self.action_space = spaces.Discrete(2)

        # Observation space depends on whether we include internal dimensions
        if include_internal_dims:
            # [opponent_last_action] + opponent_x12 + opponent_m12
            obs_dim = 1 + 2 * internal_dim
        else:
            # Just opponent's last action
            obs_dim = 1

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment.

        Returns:
            Tuple of (observations_dict, info_dict)
        """
        super().reset(seed=seed)

        self.step_count = 0
        self.last_actions = {agent: 0 for agent in self.agents}  # Default: cooperate
        self.internal_states = {
            agent: {
                "x12": np.zeros(self.internal_dim, dtype=np.float32),
                "m12": np.zeros(self.internal_dim, dtype=np.float32),
            }
            for agent in self.agents
        }

        # Statistics
        self.cooperation_count = {agent: 0 for agent in self.agents}
        self.total_rewards = {agent: 0.0 for agent in self.agents}

        observations = self._get_observations()
        info = self._get_info()

        return observations, info

    def step(
        self,
        actions: Dict[str, int],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict,
    ]:
        """
        Take a step in the environment.

        Args:
            actions: Dictionary mapping agent names to actions (0 or 1)

        Returns:
            Tuple of (observations, rewards, terminateds, truncateds, info)
        """
        self.step_count += 1

        # Store actions
        for agent in self.agents:
            self.last_actions[agent] = actions[agent]
            if actions[agent] == 0:  # Cooperate
                self.cooperation_count[agent] += 1

        # Compute rewards based on pairwise interactions
        rewards = {}
        for i, agent in enumerate(self.agents):
            # In 2-player game, compute reward against opponent
            if self.num_agents == 2:
                opponent_idx = 1 - i
                opponent_agent = self.agents[opponent_idx]
                my_action = actions[agent]
                opp_action = actions[opponent_agent]
                reward = float(self.payoff_matrix[my_action, opp_action])
            else:
                # Multi-agent: average reward against all opponents
                my_action = actions[agent]
                total_reward = 0.0
                for j, other_agent in enumerate(self.agents):
                    if i != j:
                        opp_action = actions[other_agent]
                        total_reward += self.payoff_matrix[my_action, opp_action]
                reward = total_reward / (self.num_agents - 1)

            rewards[agent] = reward
            self.total_rewards[agent] += reward

        # Check if episode is done
        done = self.step_count >= self.max_steps
        terminateds = {agent: done for agent in self.agents}
        truncateds = {agent: False for agent in self.agents}

        # Get new observations
        observations = self._get_observations()
        info = self._get_info()

        return observations, rewards, terminateds, truncateds, info

    def update_internal_states(
        self,
        internal_states: Dict[str, Dict[str, np.ndarray]],
    ):
        """
        Update internal states (x₁₂, m₁₂) for each agent.

        This should be called by the training loop after getting internal
        dimensions from the agent's forward pass.

        Args:
            internal_states: Dict mapping agent names to {'x12': array, 'm12': array}

        Example:
            >>> # After agent forward pass
            >>> env.update_internal_states({
            ...     'agent_0': {'x12': x12_0, 'm12': m12_0},
            ...     'agent_1': {'x12': x12_1, 'm12': m12_1},
            ... })
        """
        for agent, states in internal_states.items():
            if agent in self.internal_states:
                self.internal_states[agent] = {
                    "x12": np.array(states["x12"], dtype=np.float32).flatten(),
                    "m12": np.array(states["m12"], dtype=np.float32).flatten(),
                }

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents."""
        observations = {}

        for i, agent in enumerate(self.agents):
            if self.num_agents == 2:
                # Observe opponent
                opponent_idx = 1 - i
                opponent_agent = self.agents[opponent_idx]
                opponent_action = self.last_actions[opponent_agent]

                if self.include_internal_dims:
                    # Include opponent's internal dimensions
                    opponent_x12 = self.internal_states[opponent_agent]["x12"]
                    opponent_m12 = self.internal_states[opponent_agent]["m12"]
                    obs = np.concatenate([
                        [opponent_action],
                        opponent_x12,
                        opponent_m12,
                    ])
                else:
                    obs = np.array([opponent_action], dtype=np.float32)
            else:
                # Multi-agent: observe average of all opponents
                if self.include_internal_dims:
                    other_actions = []
                    other_x12 = []
                    other_m12 = []
                    for j, other_agent in enumerate(self.agents):
                        if i != j:
                            other_actions.append(self.last_actions[other_agent])
                            other_x12.append(self.internal_states[other_agent]["x12"])
                            other_m12.append(self.internal_states[other_agent]["m12"])

                    obs = np.concatenate([
                        [np.mean(other_actions)],
                        np.mean(other_x12, axis=0),
                        np.mean(other_m12, axis=0),
                    ])
                else:
                    other_actions = [
                        self.last_actions[other_agent]
                        for j, other_agent in enumerate(self.agents)
                        if i != j
                    ]
                    obs = np.array([np.mean(other_actions)], dtype=np.float32)

            observations[agent] = obs.astype(np.float32)

        return observations

    def _get_info(self) -> Dict:
        """Get info dictionary."""
        cooperation_rate = {
            agent: (
                self.cooperation_count[agent] / self.step_count
                if self.step_count > 0
                else 0.0
            )
            for agent in self.agents
        }

        return {
            "step": self.step_count,
            "cooperation_count": self.cooperation_count.copy(),
            "cooperation_rate": cooperation_rate,
            "total_rewards": self.total_rewards.copy(),
            "last_actions": self.last_actions.copy(),
        }

    def render(self):
        """Render the environment state."""
        if self.step_count == 0:
            return

        print(f"\n--- Step {self.step_count} ---")
        for agent in self.agents:
            action_str = "Cooperate" if self.last_actions[agent] == 0 else "Defect"
            print(f"{agent}: {action_str} (reward: {self.total_rewards[agent]:.1f})")

        print(f"Cooperation rates: {self._get_info()['cooperation_rate']}")


class IteratedPrisonersDilemma(PrisonersDilemma):
    """
    Iterated Prisoner's Dilemma with extended observation history.

    Extends PrisonersDilemma to track history of opponent actions and
    internal states over multiple rounds.

    Observation Space:
        [opponent_action_history (window_size),
         opponent_x12_history (window_size * internal_dim),
         opponent_m12_history (window_size * internal_dim)]

    Example:
        >>> env = IteratedPrisonersDilemma(
        ...     num_agents=2,
        ...     internal_dim=12,
        ...     max_steps=100,
        ...     history_window=5,
        ... )
    """

    def __init__(
        self,
        num_agents: int = 2,
        internal_dim: int = 12,
        max_steps: int = 100,
        history_window: int = 5,
        include_internal_dims: bool = True,
    ):
        """
        Initialize Iterated Prisoner's Dilemma.

        Args:
            num_agents: Number of agents
            internal_dim: Dimension of x₁₂ and m₁₂
            max_steps: Maximum steps per episode
            history_window: Number of past steps to include in observation
            include_internal_dims: Include opponent's x₁₂/m₁₂ in observation
        """
        self.history_window = history_window
        super().__init__(num_agents, internal_dim, max_steps, include_internal_dims)

        # Update observation space for history
        if include_internal_dims:
            obs_dim = history_window * (1 + 2 * internal_dim)
        else:
            obs_dim = history_window

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset environment and history."""
        obs, info = super().reset(seed=seed, options=options)

        # Initialize history buffers
        self.action_history = {agent: [] for agent in self.agents}
        self.x12_history = {agent: [] for agent in self.agents}
        self.m12_history = {agent: [] for agent in self.agents}

        return self._get_observations_with_history(), info

    def step(
        self,
        actions: Dict[str, int],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict,
    ]:
        """Step and update history."""
        obs, rewards, terminateds, truncateds, info = super().step(actions)

        # Update history
        for agent in self.agents:
            self.action_history[agent].append(self.last_actions[agent])
            self.x12_history[agent].append(self.internal_states[agent]["x12"].copy())
            self.m12_history[agent].append(self.internal_states[agent]["m12"].copy())

            # Trim history to window size
            if len(self.action_history[agent]) > self.history_window:
                self.action_history[agent].pop(0)
                self.x12_history[agent].pop(0)
                self.m12_history[agent].pop(0)

        observations = self._get_observations_with_history()
        return observations, rewards, terminateds, truncateds, info

    def _get_observations_with_history(self) -> Dict[str, np.ndarray]:
        """Get observations including history window."""
        observations = {}

        for i, agent in enumerate(self.agents):
            if self.num_agents == 2:
                opponent_idx = 1 - i
                opponent_agent = self.agents[opponent_idx]

                # Get opponent history (pad with zeros if not enough history)
                opp_actions = self.action_history[opponent_agent]
                opp_x12 = self.x12_history[opponent_agent]
                opp_m12 = self.m12_history[opponent_agent]

                # Pad if necessary
                pad_length = self.history_window - len(opp_actions)
                if pad_length > 0:
                    opp_actions = [0] * pad_length + opp_actions
                    opp_x12 = (
                        [np.zeros(self.internal_dim)] * pad_length + opp_x12
                    )
                    opp_m12 = (
                        [np.zeros(self.internal_dim)] * pad_length + opp_m12
                    )

                if self.include_internal_dims:
                    obs = np.concatenate([
                        np.array(opp_actions, dtype=np.float32),
                        np.concatenate(opp_x12),
                        np.concatenate(opp_m12),
                    ])
                else:
                    obs = np.array(opp_actions, dtype=np.float32)
            else:
                # Multi-agent: average history
                all_actions = []
                all_x12 = []
                all_m12 = []

                for j, other_agent in enumerate(self.agents):
                    if i != j:
                        opp_actions = self.action_history[other_agent]
                        pad_length = self.history_window - len(opp_actions)
                        if pad_length > 0:
                            opp_actions = [0] * pad_length + opp_actions
                        all_actions.append(opp_actions)

                        if self.include_internal_dims:
                            opp_x12 = self.x12_history[other_agent]
                            opp_m12 = self.m12_history[other_agent]
                            if pad_length > 0:
                                opp_x12 = (
                                    [np.zeros(self.internal_dim)] * pad_length
                                    + opp_x12
                                )
                                opp_m12 = (
                                    [np.zeros(self.internal_dim)] * pad_length
                                    + opp_m12
                                )
                            all_x12.append(opp_x12)
                            all_m12.append(opp_m12)

                avg_actions = np.mean(all_actions, axis=0)

                if self.include_internal_dims:
                    avg_x12 = np.mean(
                        [np.concatenate(x12) for x12 in all_x12], axis=0
                    )
                    avg_m12 = np.mean(
                        [np.concatenate(m12) for m12 in all_m12], axis=0
                    )
                    obs = np.concatenate([avg_actions, avg_x12, avg_m12])
                else:
                    obs = avg_actions

            observations[agent] = obs.astype(np.float32)

        return observations


class SocialGridWorld(gym.Env):
    """
    Multi-agent GridWorld with cooperative tasks.

    Multiple agents navigate a grid to collect rewards. Agents can observe
    each other's positions and internal dimensions. Tasks require cooperation
    (e.g., both agents must be at goal locations simultaneously).

    Example:
        >>> env = SocialGridWorld(size=8, num_agents=2, cooperative=True)
        >>> obs, info = env.reset()
        >>> # Each agent observes: [my_pos, other_pos, other_x12, other_m12]
    """

    def __init__(
        self,
        size: int = 8,
        num_agents: int = 2,
        internal_dim: int = 12,
        max_steps: int = 100,
        cooperative: bool = True,
        communication: bool = False,
    ):
        """
        Initialize Social GridWorld.

        Args:
            size: Grid size (size x size)
            num_agents: Number of agents
            internal_dim: Dimension of x₁₂/m₁₂
            max_steps: Maximum steps per episode
            cooperative: Require cooperation for rewards
            communication: Enable direct communication channel
        """
        super().__init__()

        self.size = size
        self.num_agents = num_agents
        self.internal_dim = internal_dim
        self.max_steps = max_steps
        self.cooperative = cooperative
        self.communication = communication

        self.agents = [f"agent_{i}" for i in range(num_agents)]

        # Actions: 0=up, 1=right, 2=down, 3=left, 4=stay
        self.action_space = spaces.Discrete(5)

        # Observation: [my_x, my_y, goal_x, goal_y] + other agents + internal dims
        # For each other agent: [other_x, other_y, x12..., m12...]
        obs_dim = 4 + (num_agents - 1) * (2 + 2 * internal_dim)
        self.observation_space = spaces.Box(
            low=0,
            high=max(size, 1),
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        self.step_count = 0

        # Random positions for agents
        self.positions = {}
        for agent in self.agents:
            self.positions[agent] = self.np_random.integers(0, self.size, size=2)

        # Random goal positions
        self.goals = {}
        for agent in self.agents:
            self.goals[agent] = self.np_random.integers(0, self.size, size=2)

        # Internal states
        self.internal_states = {
            agent: {
                "x12": np.zeros(self.internal_dim, dtype=np.float32),
                "m12": np.zeros(self.internal_dim, dtype=np.float32),
            }
            for agent in self.agents
        }

        # Statistics
        self.total_rewards = {agent: 0.0 for agent in self.agents}
        self.goals_reached = {agent: 0 for agent in self.agents}

        observations = self._get_observations()
        info = self._get_info()

        return observations, info

    def step(
        self,
        actions: Dict[str, int],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict,
    ]:
        """Take a step in the environment."""
        self.step_count += 1

        # Move agents
        for agent, action in actions.items():
            pos = self.positions[agent]
            if action == 0:  # up
                pos[1] = max(0, pos[1] - 1)
            elif action == 1:  # right
                pos[0] = min(self.size - 1, pos[0] + 1)
            elif action == 2:  # down
                pos[1] = min(self.size - 1, pos[1] + 1)
            elif action == 3:  # left
                pos[0] = max(0, pos[0] - 1)
            # action == 4: stay

        # Compute rewards
        rewards = {}
        all_at_goal = True

        for agent in self.agents:
            at_goal = np.array_equal(self.positions[agent], self.goals[agent])
            if at_goal:
                self.goals_reached[agent] += 1
            else:
                all_at_goal = False

            if self.cooperative:
                # Reward only if all agents at goal
                reward = 10.0 if all_at_goal else -0.1
            else:
                # Individual reward
                reward = 10.0 if at_goal else -0.1

            rewards[agent] = reward
            self.total_rewards[agent] += reward

        # Check termination
        done = self.step_count >= self.max_steps
        terminateds = {agent: done for agent in self.agents}
        truncateds = {agent: False for agent in self.agents}

        observations = self._get_observations()
        info = self._get_info()

        return observations, rewards, terminateds, truncateds, info

    def update_internal_states(
        self,
        internal_states: Dict[str, Dict[str, np.ndarray]],
    ):
        """Update internal states (x₁₂, m₁₂) for each agent."""
        for agent, states in internal_states.items():
            if agent in self.internal_states:
                self.internal_states[agent] = {
                    "x12": np.array(states["x12"], dtype=np.float32).flatten(),
                    "m12": np.array(states["m12"], dtype=np.float32).flatten(),
                }

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents."""
        observations = {}

        for i, agent in enumerate(self.agents):
            my_pos = self.positions[agent]
            my_goal = self.goals[agent]

            # Start with my position and goal
            obs_parts = [my_pos, my_goal]

            # Add other agents' information
            for j, other_agent in enumerate(self.agents):
                if i != j:
                    other_pos = self.positions[other_agent]
                    other_x12 = self.internal_states[other_agent]["x12"]
                    other_m12 = self.internal_states[other_agent]["m12"]

                    obs_parts.extend([other_pos, other_x12, other_m12])

            observations[agent] = np.concatenate(obs_parts).astype(np.float32)

        return observations

    def _get_info(self) -> Dict:
        """Get info dictionary."""
        return {
            "step": self.step_count,
            "positions": self.positions.copy(),
            "goals": self.goals.copy(),
            "goals_reached": self.goals_reached.copy(),
            "total_rewards": self.total_rewards.copy(),
        }

    def render(self):
        """Render the grid."""
        grid = np.zeros((self.size, self.size), dtype=str)
        grid[:] = '.'

        # Place goals
        for i, agent in enumerate(self.agents):
            goal = self.goals[agent]
            grid[goal[1], goal[0]] = f'G{i}'

        # Place agents
        for i, agent in enumerate(self.agents):
            pos = self.positions[agent]
            grid[pos[1], pos[0]] = f'A{i}'

        print(f"\n--- Step {self.step_count} ---")
        for row in grid:
            print(' '.join(row))
        print(f"Total rewards: {self.total_rewards}")

"""
GridWorld Environment for Internal Dimension AI

A configurable gridworld environment with:
- Configurable size, obstacles, rewards
- Sparse or dense rewards
- Moving goals
- Trap mechanics (for wisdom testing)
- State rendering/visualization
- Gymnasium API compatible
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any, List
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import io
from PIL import Image


class GridWorld(gym.Env):
    """
    GridWorld environment for testing internal dimensions.

    Features:
    - Configurable grid size
    - Obstacles (walls)
    - Rewards (goal states)
    - Traps (negative reward, discovered once)
    - Sparse or dense rewards
    - Moving goal support
    - ASCII and matplotlib rendering
    """

    metadata = {'render_modes': ['human', 'rgb_array', 'ascii'], 'render_fps': 4}

    def __init__(
        self,
        size: int = 10,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        goal_position: Optional[Tuple[int, int]] = None,
        trap_positions: Optional[List[Tuple[int, int]]] = None,
        reward_goal: float = 1.0,
        reward_step: float = -0.01,
        reward_trap: float = -0.5,
        reward_wall: float = -0.1,
        sparse_rewards: bool = False,
        moving_goal: bool = False,
        goal_move_interval: int = 20,
        random_obstacles: bool = False,
        obstacle_density: float = 0.1,
        render_mode: Optional[str] = None
    ):
        """
        Initialize GridWorld.

        Args:
            size: Grid size (size x size)
            obstacles: List of obstacle positions (x, y)
            goal_position: Goal position (x, y)
            trap_positions: List of trap positions
            reward_goal: Reward for reaching goal
            reward_step: Reward for each step
            reward_trap: Reward for stepping on trap
            reward_wall: Reward for hitting wall
            sparse_rewards: If True, only give reward at goal
            moving_goal: Whether goal moves periodically
            goal_move_interval: Steps between goal movements
            random_obstacles: Generate random obstacles
            obstacle_density: Density of random obstacles [0, 1]
            render_mode: Rendering mode ('human', 'rgb_array', or 'ascii')
        """
        super().__init__()

        self.size = size
        self.reward_goal = reward_goal
        self.reward_step = reward_step if not sparse_rewards else 0.0
        self.reward_trap = reward_trap
        self.reward_wall = reward_wall
        self.sparse_rewards = sparse_rewards
        self.moving_goal = moving_goal
        self.goal_move_interval = goal_move_interval
        self.render_mode = render_mode

        # Action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)

        # Observation space: (x, y) position
        self.observation_space = spaces.Box(
            low=0,
            high=size - 1,
            shape=(2,),
            dtype=np.float32
        )

        # Initialize obstacles
        if random_obstacles:
            self.obstacles = self._generate_random_obstacles(obstacle_density)
        elif obstacles is not None:
            self.obstacles = set(obstacles)
        else:
            self.obstacles = set()

        # Initialize goal
        if goal_position is not None:
            self.goal_position = goal_position
        else:
            self.goal_position = self._find_empty_position()

        # Initialize traps
        if trap_positions is not None:
            self.trap_positions = set(trap_positions)
        else:
            self.trap_positions = set()

        # Trap discovery tracking (for wisdom: learn from mistakes)
        self.discovered_traps = set()

        # State
        self.agent_position = None
        self.steps = 0
        self.episode_reward = 0

        # Visualization
        self.fig = None
        self.ax = None

    def _generate_random_obstacles(self, density: float) -> set:
        """Generate random obstacles."""
        obstacles = set()
        num_obstacles = int(self.size * self.size * density)

        while len(obstacles) < num_obstacles:
            pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            # Don't place obstacles at start or goal
            if pos != (0, 0) and pos != self.goal_position:
                obstacles.add(pos)

        return obstacles

    def _find_empty_position(self) -> Tuple[int, int]:
        """Find a random empty position."""
        while True:
            pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            if pos not in self.obstacles and pos != (0, 0):
                return pos

    def _move_goal(self):
        """Move goal to a new random position."""
        self.goal_position = self._find_empty_position()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)

        # Reset agent position to start (0, 0) or random
        if options and options.get('random_start', False):
            self.agent_position = self._find_empty_position()
        else:
            self.agent_position = (0, 0)

        # Reset discovered traps
        self.discovered_traps = set()

        self.steps = 0
        self.episode_reward = 0

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        return np.array(self.agent_position, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: 0=up, 1=right, 2=down, 3=left

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.steps += 1

        # Move goal if enabled
        if self.moving_goal and self.steps % self.goal_move_interval == 0:
            self._move_goal()

        # Compute new position
        x, y = self.agent_position

        if action == 0:  # up
            new_position = (x, y - 1)
        elif action == 1:  # right
            new_position = (x + 1, y)
        elif action == 2:  # down
            new_position = (x, y + 1)
        elif action == 3:  # left
            new_position = (x - 1, y)
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check bounds
        new_x, new_y = new_position
        if new_x < 0 or new_x >= self.size or new_y < 0 or new_y >= self.size:
            # Hit wall
            reward = self.reward_wall
            terminated = False
            truncated = False
        elif new_position in self.obstacles:
            # Hit obstacle
            reward = self.reward_wall
            terminated = False
            truncated = False
        else:
            # Valid move
            self.agent_position = new_position

            # Check for goal
            if self.agent_position == self.goal_position:
                reward = self.reward_goal
                terminated = True
                truncated = False
            # Check for traps
            elif self.agent_position in self.trap_positions:
                # Only give negative reward if trap not yet discovered
                if self.agent_position not in self.discovered_traps:
                    reward = self.reward_trap
                    self.discovered_traps.add(self.agent_position)
                else:
                    # Trap already discovered, just step penalty
                    reward = self.reward_step

                terminated = False
                truncated = False
            else:
                # Normal step
                reward = self.reward_step
                terminated = False
                truncated = False

        self.episode_reward += reward

        # Check for max steps (truncation)
        max_steps = self.size * self.size * 4  # Heuristic
        if self.steps >= max_steps:
            truncated = True

        info = {
            'steps': self.steps,
            'episode_reward': self.episode_reward,
            'discovered_traps': len(self.discovered_traps),
            'total_traps': len(self.trap_positions)
        }

        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == 'ascii':
            return self._render_ascii()
        elif self.render_mode == 'rgb_array':
            return self._render_rgb()
        elif self.render_mode == 'human':
            return self._render_matplotlib()
        else:
            return None

    def _render_ascii(self) -> str:
        """Render as ASCII art."""
        grid = [[' ' for _ in range(self.size)] for _ in range(self.size)]

        # Place obstacles
        for obs_x, obs_y in self.obstacles:
            grid[obs_y][obs_x] = '#'

        # Place traps
        for trap_x, trap_y in self.trap_positions:
            if (trap_x, trap_y) in self.discovered_traps:
                grid[trap_y][trap_x] = 'X'  # Discovered trap
            else:
                grid[trap_y][trap_x] = ' '  # Hidden trap

        # Place goal
        goal_x, goal_y = self.goal_position
        grid[goal_y][goal_x] = 'G'

        # Place agent
        agent_x, agent_y = self.agent_position
        grid[agent_y][agent_x] = 'A'

        # Create ASCII string
        result = '+' + '-' * self.size + '+\n'
        for row in grid:
            result += '|' + ''.join(row) + '|\n'
        result += '+' + '-' * self.size + '+\n'

        return result

    def _render_matplotlib(self):
        """Render using matplotlib."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.ax.clear()
        self.ax.set_xlim(-0.5, self.size - 0.5)
        self.ax.set_ylim(-0.5, self.size - 0.5)
        self.ax.set_aspect('equal')
        self.ax.set_xticks(range(self.size))
        self.ax.set_yticks(range(self.size))
        self.ax.grid(True, alpha=0.3)
        self.ax.invert_yaxis()

        # Draw obstacles
        for obs_x, obs_y in self.obstacles:
            rect = Rectangle((obs_x - 0.5, obs_y - 0.5), 1, 1,
                           facecolor='black', edgecolor='black')
            self.ax.add_patch(rect)

        # Draw discovered traps
        for trap_x, trap_y in self.discovered_traps:
            rect = Rectangle((trap_x - 0.5, trap_y - 0.5), 1, 1,
                           facecolor='red', alpha=0.5, edgecolor='darkred')
            self.ax.add_patch(rect)
            self.ax.text(trap_x, trap_y, 'X', ha='center', va='center',
                       fontsize=20, color='darkred', weight='bold')

        # Draw undiscovered traps (hidden)
        for trap_x, trap_y in self.trap_positions:
            if (trap_x, trap_y) not in self.discovered_traps:
                # Faint outline (player doesn't see this in reality)
                rect = Rectangle((trap_x - 0.5, trap_y - 0.5), 1, 1,
                               facecolor='none', edgecolor='red',
                               alpha=0.2, linestyle='--')
                self.ax.add_patch(rect)

        # Draw goal
        goal_x, goal_y = self.goal_position
        rect = Rectangle((goal_x - 0.5, goal_y - 0.5), 1, 1,
                       facecolor='gold', alpha=0.7, edgecolor='orange')
        self.ax.add_patch(rect)
        self.ax.text(goal_x, goal_y, 'G', ha='center', va='center',
                   fontsize=20, color='orange', weight='bold')

        # Draw agent
        agent_x, agent_y = self.agent_position
        circle = plt.Circle((agent_x, agent_y), 0.3, color='blue', zorder=10)
        self.ax.add_patch(circle)

        self.ax.set_title(f'GridWorld (Steps: {self.steps}, Reward: {self.episode_reward:.2f})')

        plt.pause(0.001)
        return self.fig

    def _render_rgb(self) -> np.ndarray:
        """Render as RGB array."""
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()

        # Draw grid (similar to matplotlib render)
        for obs_x, obs_y in self.obstacles:
            rect = Rectangle((obs_x - 0.5, obs_y - 0.5), 1, 1,
                           facecolor='black')
            ax.add_patch(rect)

        for trap_x, trap_y in self.discovered_traps:
            rect = Rectangle((trap_x - 0.5, trap_y - 0.5), 1, 1,
                           facecolor='red', alpha=0.5)
            ax.add_patch(rect)

        goal_x, goal_y = self.goal_position
        rect = Rectangle((goal_x - 0.5, goal_y - 0.5), 1, 1,
                       facecolor='gold', alpha=0.7)
        ax.add_patch(rect)

        agent_x, agent_y = self.agent_position
        circle = plt.Circle((agent_x, agent_y), 0.3, color='blue')
        ax.add_patch(circle)

        # Convert to RGB array
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return data

    def close(self):
        """Close the environment."""
        if self.fig is not None:
            plt.close(self.fig)


class TwoRoomGridWorld(gym.Env):
    """
    Two-room GridWorld for curiosity testing.

    Layout:
        ┌──────────┬──────────┐
        │          │          │
        │  Room A  │  Room B  │
        │(Familiar)│ (Novel)  │
        │          │          │
        └──────────┴──────────┘

    Agent starts in Room A. Room B contains novel features.
    Used to test curiosity-driven exploration.
    """

    metadata = {'render_modes': ['human', 'rgb_array', 'ascii'], 'render_fps': 4}

    def __init__(
        self,
        room_size: int = 5,
        door_position: int = 2,
        novel_features: bool = True,
        render_mode: Optional[str] = None
    ):
        """
        Initialize two-room environment.

        Args:
            room_size: Size of each room
            door_position: Y-position of door between rooms
            novel_features: Whether to add novel features to Room B
            render_mode: Rendering mode
        """
        super().__init__()

        self.room_size = room_size
        self.total_width = room_size * 2
        self.door_position = door_position
        self.novel_features = novel_features
        self.render_mode = render_mode

        # Action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)

        # Observation space
        self.observation_space = spaces.Box(
            low=0,
            high=self.total_width - 1,
            shape=(2,),
            dtype=np.float32
        )

        # Build walls
        self.walls = set()
        for y in range(room_size):
            if y != door_position:  # Leave door open
                self.walls.add((room_size, y))

        # Novel features in Room B (optional rewards)
        self.novel_feature_positions = set()
        if novel_features:
            self.novel_feature_positions = {
                (room_size + 2, 2),
                (room_size + 3, 3),
            }

        # State
        self.agent_position = None
        self.steps = 0
        self.visited_room_b = False

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)

        # Start in Room A
        self.agent_position = (1, 1)
        self.steps = 0
        self.visited_room_b = False

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Get observation."""
        return np.array(self.agent_position, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step."""
        self.steps += 1

        x, y = self.agent_position

        if action == 0:  # up
            new_position = (x, y - 1)
        elif action == 1:  # right
            new_position = (x + 1, y)
        elif action == 2:  # down
            new_position = (x, y + 1)
        elif action == 3:  # left
            new_position = (x - 1, y)
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check bounds and walls
        new_x, new_y = new_position
        if (new_x < 0 or new_x >= self.total_width or
            new_y < 0 or new_y >= self.room_size or
            new_position in self.walls):
            # Invalid move, stay in place
            reward = -0.01
        else:
            # Valid move
            self.agent_position = new_position

            # Check if entered Room B
            if new_x >= self.room_size:
                self.visited_room_b = True

            # Small negative step reward (encourage curiosity, not aimless wandering)
            reward = -0.01

            # Bonus for discovering novel features
            if new_position in self.novel_feature_positions:
                reward += 0.5

        terminated = False
        truncated = self.steps >= 100

        info = {
            'steps': self.steps,
            'visited_room_b': self.visited_room_b,
            'in_room_b': self.agent_position[0] >= self.room_size
        }

        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == 'ascii':
            return self._render_ascii()
        return None

    def _render_ascii(self) -> str:
        """ASCII rendering."""
        grid = [[' ' for _ in range(self.total_width)] for _ in range(self.room_size)]

        # Draw walls
        for wall_x, wall_y in self.walls:
            grid[wall_y][wall_x] = '|'

        # Draw novel features
        for feat_x, feat_y in self.novel_feature_positions:
            grid[feat_y][feat_x] = '*'

        # Draw agent
        agent_x, agent_y = self.agent_position
        grid[agent_y][agent_x] = 'A'

        # Create ASCII
        result = '+' + '-' * self.total_width + '+\n'
        for row in grid:
            result += '|' + ''.join(row) + '|\n'
        result += '+' + '-' * self.total_width + '+\n'
        result += f"Room A: 0-{self.room_size-1}, Room B: {self.room_size}-{self.total_width-1}\n"
        result += f"Visited Room B: {self.visited_room_b}\n"

        return result

    def close(self):
        """Close environment."""
        pass

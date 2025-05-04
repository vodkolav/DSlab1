# telemetry/telemetry_manager.py
from collections import defaultdict
import numpy as np
import json

class TelemetryManager:
    """
    Manages the collection of training and evaluation metrics.
    """
    def __init__(self):
        # Use defaultdicts to store lists of metrics per episode/step
        self.episode_rewards = []
        self.episode_lengths = []
        self.cumulative_rewards = [] # Optional: Cumulative reward over training
        self.algorithm_specific_metrics = defaultdict(list) # For things like TD error, policy change

        self._current_episode_reward = 0
        self._current_episode_length = 0

    def reset_episode_metrics(self):
        """Resets metrics for a new episode."""
        self._current_episode_reward = 0
        self._current_episode_length = 0

    def record_step(self, reward: float, info: dict = None):
        """Records metrics for a single step."""
        self._current_episode_reward += reward
        self._current_episode_length += 1
        # You can record other step-specific info if needed from the 'info' dict

    def record_episode_end(self, total_timesteps_so_far: int):
        """Records metrics at the end of an episode."""
        self.episode_rewards.append(self._current_episode_reward)
        self.episode_lengths.append(self._current_episode_length)
        # Calculate and record cumulative reward
        if not self.cumulative_rewards:
            self.cumulative_rewards.append(self._current_episode_reward)
        else:
            self.cumulative_rewards.append(self.cumulative_rewards[-1] + self._current_episode_reward)

        self.reset_episode_metrics() # Prepare for the next episode

    def record_algorithm_metric(self, metric_name: str, value):
        """Records algorithm-specific metrics (e.g., delta in Value Iteration)."""
        self.algorithm_specific_metrics[metric_name].append(value)

    def get_average_reward(self, window_size: int = 100) -> float:
        """Calculates the average reward over the last window_size episodes."""
        if not self.episode_rewards:
            return 0.0
        return np.mean(self.episode_rewards[-window_size:])

    def get_total_episodes(self) -> int:
        """Returns the total number of recorded episodes."""
        return len(self.episode_rewards)

    def save_metrics(self, filename: str):
        """Saves collected metrics to a JSON file."""
        metrics_data = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "cumulative_rewards": self.cumulative_rewards,
            "algorithm_specific_metrics": self.algorithm_specific_metrics
        }
        with open(filename, 'w') as f:
            json.dump(metrics_data, f)
        print(f"Metrics saved to {filename}")

    def load_metrics(self, filename: str):
        """Loads metrics from a JSON file."""
        try:
            with open(filename, 'r') as f:
                metrics_data = json.load(f)
                self.episode_rewards = metrics_data.get("episode_rewards", [])
                self.episode_lengths = metrics_data.get("episode_lengths", [])
                self.cumulative_rewards = metrics_data.get("cumulative_rewards", [])
                self.algorithm_specific_metrics = defaultdict(list, metrics_data.get("algorithm_specific_metrics", {}))
            print(f"Metrics loaded from {filename}")
        except FileNotFoundError:
            print(f"Error: Metrics file not found at {filename}")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {filename}")
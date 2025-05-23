# telemetry/telemetry_manager.py
from collections import defaultdict
import numpy as np
import json
from datetime import datetime
from copy import deepcopy

class TelemetryManager:
    """
    Manages the collection of training and evaluation metrics.
    """
    def __init__(self, env, algorithm):
        
        self.metadata = {
            "start_time": datetime.now().strftime(r"%y.%m.%d-%H.%M"),  
            "env_name": env.spec.id if env.spec else "Unknown",
            "algorithm": algorithm.get_parameters()
        }

        # Use defaultdicts to store lists of metrics per episode/step
        self.episodes = []
        self.cumulative_rewards = [] # Optional: Cumulative reward over training
        self.algorithm_specific_metrics = defaultdict(list) # For things like TD error, policy change

        self.reset_episode_metrics()

    def reset_episode_metrics(self):
        """Resets metrics for a new episode."""
        self._current_episode = { 
            "reward": 0,
            "length": 0,
            "replay": [],
            "info": [],
            "internal_state":{}  
        } 

    def record_step(self, reward: float, info: dict = None, frame=None):
        """Records metrics for a single step."""
        self._current_episode["reward"] += reward
        self._current_episode["length"] += 1
        self._current_episode["replay"].append(frame)

        #self._current_episode["info"].append(info if info is not None else {})
        # You can record other step-specific info if needed from the 'info' dict

    def record_episode_end(self, internal_state, is_training = True):
        """Records metrics at the end of an episode."""
        self._current_episode["is_training"] = is_training
        self._current_episode["n"] = self.get_total_episodes() + 1
        self._current_episode["internal_state"] = dict(internal_state)

        self.episodes.append(deepcopy(self._current_episode))
        
        # Calculate and record cumulative reward
        if not self.cumulative_rewards:
            self.cumulative_rewards.append(self._current_episode["reward"])
        else:
            self.cumulative_rewards.append(self.cumulative_rewards[-1] + self._current_episode["reward"])

        self.reset_episode_metrics() # Prepare for the next episode

    def record_algorithm_metric(self, metric_name: str, value):
        """Records algorithm-specific metrics (e.g., delta in Value Iteration)."""
        self.algorithm_specific_metrics[metric_name].append(value)

    def get_average_reward(self, window_size: int = 100) -> float:
        """Calculates the average reward over the last window_size episodes."""
        if not self.episodes:
            return 0.0
        return np.mean(self.episodes["reward"][-window_size:])

    def get_total_episodes(self) -> int:
        """Returns the total number of recorded episodes."""
        return len(self.episodes)

    def save_metrics(self, filename: str):
        """Saves collected metrics to a JSON file."""
        metrics_data = {
            "metadata": self.metadata,
            "episodes": self.episodes,  
            "cumulative_rewards": self.cumulative_rewards,
            "algorithm_specific_metrics": self.algorithm_specific_metrics
        }
        with open(filename, 'w') as f:
            json.dump(metrics_data, f, indent=4)
        print(f"Metrics saved to {filename}")

    def load_metrics(self, filename: str):
        """Loads metrics from a JSON file."""
        try:
            with open(filename, 'r') as f:
                metrics_data = json.load(f)
                self.metadata = metrics_data.get("metadata", {}) 
                self.episodes = metrics_data.get("episodes", [])
                self.cumulative_rewards = metrics_data.get("cumulative_rewards",0) ,
                self.algorithm_specific_metrics = defaultdict(list, metrics_data.get("algorithm_specific_metrics", {}))
            print(f"Metrics loaded from {filename}")
        except FileNotFoundError:
            print(f"Error: Metrics file not found at {filename}")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {filename}")
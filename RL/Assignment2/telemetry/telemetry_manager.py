# telemetry/telemetry_manager.py
from collections import defaultdict
import numpy as np
import json
from datetime import datetime
import time
import psutil
import os
from copy import deepcopy


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types 
        taken from https://stackoverflow.com/a/49677241/7097017
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
class TelemetryManager:
    """
    Manages the collection of training and evaluation metrics.
    """
    def __init__(self, env, algorithm):
        
        self.metadata = {
            "start_time": datetime.now().strftime(r"%y.%m.%d-%H.%M"),  
            "env_name": env.spec.id if env.spec else "Unknown",
            "algorithm": algorithm.get_parameters(),
            "rss_mb": self.get_memory_usage_mb()
        }

        # Use defaultdicts to store lists of metrics per episode/step
        self.episodes = []
        #self.algorithm_specific_metrics = dict(list) # For things like TD error, policy change

        self.reset_episode_metrics()

    def reset_episode_metrics(self):
        """Resets metrics for a new episode."""
        self._current_episode = { 
            "reward": 0,
            "length": 0,
            "replay": [],
            "info": [],
            "internal_state":{},  
            "n": self.get_total_episodes() + 1,
            "start": time.time()
        } 

    def record_step(self, reward: float, info: dict = None, frame=None):
        """Records metrics for a single step."""
        self._current_episode["reward"] += reward
        self._current_episode["length"] += 1
        #self._current_episode["replay"].append(frame)

        #self._current_episode["info"].append(info if info is not None else {})
        # You can record other step-specific info if needed from the 'info' dict

    def record_episode_end(self, agent , is_training = True):
        """Records metrics at the end of an episode."""
        self._current_episode["is_training"] = is_training

        self._current_episode["end"] = time.time()

        self._current_episode["internal_state"] = dict(agent.get_intestines())

        self._current_episode["rss_mb"] = self.get_memory_usage_mb()

        self.episodes.append(deepcopy(self._current_episode))

        self.reset_episode_metrics() # Prepare for the next episode


    def report_start(self, is_training, num_episodes):
        self.total_episodes = num_episodes
        self.mode = "Training" if is_training else "Evaluating"
        print(f"{self.mode} {self.metadata['algorithm']['algorithm']} over {self.total_episodes} episodes")


    def report_progress(self):
        # Optional: Print progress
        freq = 10  # Frequency of reporting progress
        i_episode = self._current_episode["n"]
        if (i_episode + 1) % freq == 0:
            avg_reward = self.get_average_reward()
            print(f"\r {self.mode} Episode {i_episode + 1}/{self.total_episodes}, Avg Reward (last 100): {avg_reward:.2f}", end='')


    def report_end(self):
        # Optional: Print end message
        print(f"\n{self.mode} ended. Total episodes recorded: {len(self.episodes)}")


    def get_memory_usage_mb(self):
        """
        Returns the current process's resident set size (RSS) memory usage in MB using psutil.
        Works cross-platform.
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024) # RSS in bytes
    

    def get_average_reward(self, window_size: int = 100) -> float:
        """Calculates the average reward over the last window_size episodes."""
        if not self.episodes:
            return 0.0
        return np.mean([ep["reward"] for ep in self.episodes[-window_size:]])

    def get_total_episodes(self) -> int:
        """Returns the total number of recorded episodes."""
        return len(self.episodes)

    def save_metrics(self, filename: str):
        """Saves collected metrics to a JSON file."""
        metrics_data = {
            "metadata": self.metadata,
            "episodes": self.episodes,  
        }
        with open(filename, 'w') as f:
            json.dump(metrics_data, f, indent=4, cls=NumpyEncoder)
        print(f"Metrics saved to {filename}")

    def load_metrics(self, filename: str):
        """Loads metrics from a JSON file."""
        try:
            with open(filename, 'r') as f:
                metrics_data = json.load(f)
                self.metadata = metrics_data.get("metadata", {}) 
                self.episodes = metrics_data.get("episodes", [])
            print(f"Metrics loaded from {filename}")
        except FileNotFoundError:
            print(f"Error: Metrics file not found at {filename}")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {filename}")
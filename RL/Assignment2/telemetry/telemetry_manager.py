# telemetry/telemetry_manager.py
from collections import defaultdict
import numpy as np
import json
from datetime import datetime
import time
import psutil
import os
from copy import deepcopy
from algorithms.agent import RLAgent

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
    def __init__(self, env, algorithm, limit = 100):
        
        self.metadata = {
            "start_time": datetime.now().strftime(r"%y.%m.%d-%H.%M"),  
            "env_name": env.spec.id if env.spec else "Unknown",
            "rss_mb": self.get_memory_usage_mb()
        }
        self.metadata.update(algorithm.get_parameters())
    
        self.tot_episodes = 0

        self.samplePoints = list(range(limit))

        # Use defaultdicts to store lists of metrics per episode/step
        self.episodes = []
        #self.algorithm_specific_metrics = dict(list) # For things like TD error, policy change

        self.reset_episode_metrics(0)

    @property
    def total_episodes(self):
        return self.tot_episodes

    @total_episodes.setter
    def total_episodes(self, value):
        limit = len(self.samplePoints)
        if value < limit:
            limit = value
        
        self.samplePoints = np.int64(np.linspace(0,value, limit))
        invl = np.round(value/limit, decimals=2)
        print("tracking and reporting once every", invl, "episodes")
        self.tot_episodes = value

    @property
    def mode(self):
        return "Training" if self._current_episode["is_training"] else "Evaluating"

    def reset_episode_metrics(self, i_episode):
        """Resets metrics for a new episode."""
        self.i_episode = i_episode
        self._current_episode = { 
            "reward": 0,
            "length": 0,
            "replay": [],
            "info": [],
            "internal_state":{},  
            "i": i_episode, 
            "start": time.time()
        } 

    def record_step(self, reward: float, info: dict = None, frame=None):
        """Records metrics for a single step."""
        self._current_episode["reward"] += reward
        self._current_episode["length"] += 1
        #self._current_episode["replay"].append(frame)

        #self._current_episode["info"].append(info if info is not None else {})
        # You can record other step-specific info if needed from the 'info' dict

    def record_episode_end(self, agent: RLAgent , is_training = True):
        """Records metrics at the end of an episode."""
        if self.i_episode in self.samplePoints:

            self._current_episode["is_training"] = is_training

            self._current_episode["end"] = time.time()

            self._current_episode["internal_state"] = dict(agent.get_intestines())
            
            self._current_episode["epsilon"] = agent.strategy.epsilon

            self._current_episode["rss_mb"] = self.get_memory_usage_mb()

            self._current_episode["size_bytes"] = agent.size()

            self.episodes.append(deepcopy(self._current_episode))


    def report(self, what, newline = False):
        if newline:
            print(what)
        else:
            print(f"\r{what}" , end='')


    def start(self, num_episodes):
        # Start telemetry reporting for an experiment
        self.total_episodes = num_episodes
        run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.metadata["start_time"] = run_timestamp
        self.metadata["pid"] = os.getpid()
        self.report(f" Starting {self.metadata['name']} over {num_episodes} episodes", newline=True)


    def progress(self):
        # Optional: Print progress        
        if self.i_episode in self.samplePoints:
            avg_reward = self.get_average_reward()
            msg = f"\r {self.mode} Episode {self.i_episode}/{self.total_episodes}, Avg Reward (last 100): {avg_reward:.2f}"
            self.report(msg)


    def end(self):
        # Optional: Print end message
        end_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.metadata["end_time"] = end_timestamp
        self.report(f"\n {self.metadata['name']} ended. Total episodes recorded: {len(self.episodes)}", newline=True)


    def get_memory_usage_mb(self):
        """
        Returns the current process's resident set size (RSS) memory usage in MB using psutil.
        Works cross-platform.
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024) # RSS in megabytes
    

    def get_average_reward(self, window_size: int = 100) -> float:
        """Calculates the average reward over the last window_size episodes."""
        if not self.episodes:
            return 0.0
        return np.mean([ep["reward"] for ep in self.episodes[-window_size:]])

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
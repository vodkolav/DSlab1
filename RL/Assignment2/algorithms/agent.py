# algorithms/base_algorithm.py
import gymnasium as gym
import numpy as np
from utils.strategy import Strategy

class RLAgent:
    """
    Base class for reinforcement learning algorithms.
    Subclasses should implement the choose_action and update methods.
    """
    def __init__(self, env: gym.Env, strategy: Strategy, gamma: float = 1.0, **kwargs):
        """
        Initializes the base RL algorithm.

        Args:
            env: The Gymnasium environment.
            gamma: Discount factor.
            **kwargs: Additional algorithm-specific parameters.
        """
        if not isinstance(env, gym.Env):
            raise TypeError("env must be a Gymnasium environment.")
        if not (0 <= gamma <= 1):
             raise ValueError("gamma must be between 0 and 1.")

        self.env = env
        self.gamma = gamma
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        #strategy.n_actions = env.action_space.n
        self.strategy = strategy
        self.params = kwargs
        self.terminate_prematurely = False
        
        # Algorithm-specific state (e.g., Q-table, Value table) will be initialized in subclasses

    def get_parameters(self) -> dict:
        """
        Returns the parameters of the algorithm.
        """
        return {
            "algorithm": __class__.__name__,
            "gamma": self.gamma,
            "strategy": self.strategy.get_parameters()
        }

    def choose_action(self, state: int, episode: int) -> int:
        """
        Selects an action based on the current policy or learned values.
        Must be implemented by subclasses that perform control.

        Args:
            state: The current state observation.

        Returns:
            The chosen action.
        """
        # This method will now handle exploration strategies like epsilon-greedy internally
        # based on the algorithm's current state and possibly passed parameters (like epsilon)
        raise NotImplementedError("Subclass must implement abstract method")

    def choose_greedy_action(self, state: int) -> int:
        """
        Selects the greedy action of the strategy.
        For use in evaluation.
        Must be implemented by subclasses that perform control.

        Args:
            state: The current state observation.

        Returns:
            The chosen action.
        """
        # This method will now handle exploration strategies like epsilon-greedy internally
        # based on the algorithm's current state and possibly passed parameters (like epsilon)
        raise NotImplementedError("Subclass must implement abstract method")

    def update(self, state, action, reward, next_state, terminated, truncated, **kwargs):
         """
         Updates the algorithm's state based on a single step of experience.
         The signature might vary slightly depending on the algorithm
         (e.g., SARSA needs next_action).

         Args:
             state: The state before the action.
             action: The action taken.
             reward: The reward received.
             next_state: The state after the action.
             terminated: Whether the episode terminated.
             truncated: Whether the episode was truncated.
             **kwargs: Additional update-specific parameters (e.g., next_action for SARSA).
         """
         raise NotImplementedError("Subclass must implement abstract method")

    # Keep optional methods like get_policy, get_value_function, save, load

    def get_policy(self):
        """
        Returns the learned policy (e.g., as a numpy array or function).
        Optional method, implementation depends on the algorithm.
        """
        pass

    def get_value_function(self):
        """
        Returns the learned value function.
        Optional method, implementation depends on the algorithm.
        """
        pass

    # ... save and load methods
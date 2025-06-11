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

    def size(self):
        """
        Returns estimated size of the object
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def default_dictionary_size(self, collection):
        #TODO: move it somewhere else
        import sys

        # To get a better estimate of the contents, you'd have to iterate:
        total_q_table_size = sys.getsizeof(collection) # Base size of the defaultdict
        
        if type(collection) is list:
            for i in collection:
                total_q_table_size += sys.getsizeof(i)
        else:
            for state_key, action_values in collection.items():
                total_q_table_size += sys.getsizeof(state_key) # Size of the key
                total_q_table_size += sys.getsizeof(action_values) # Size of the numpy array object
                total_q_table_size += action_values.nbytes # Actual data in the numpy array (often larger)
        return total_q_table_size
        #print(f"Estimated total size of Q-table and its contents: {total_q_table_size} bytes")


    def get_parameters(self) -> dict:
        """
        Returns the parameters of the algorithm.
        """
        return {
            "name": self.__class__.__name__,
            "moniker": self.__class__.__name__,
            "params": {
                "gamma": self.gamma
                      }
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

    def get_intestines(self):
        """
        Returns the internal state of the algorithm (e.g., Q-table, Value table).
        Optional method, implementation depends on the algorithm.
        """
        raise NotImplementedError("Subclass must implement abstract method")

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
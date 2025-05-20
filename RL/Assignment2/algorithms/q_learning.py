# algorithms/q_learning.py
import gymnasium as gym
import numpy as np
from collections import defaultdict
from utils.strategy import Strategy
from algorithms.base_algorithm import RLAlgorithm

class QLearning(RLAlgorithm):
    """
    Q-Learning algorithm implementation.
    """
    def __init__(self, env: gym.Env, strategy: Strategy,
                 gamma: float = 1.0, alpha: float = 0.1,
                  **kwargs):
        """
        Initializes the Q-Learning algorithm.

        Args:
            env: The Gymnasium environment.
            gamma: Discount factor.
            alpha: Learning rate.
            **kwargs: Additional parameters for the base class.
        """
        super().__init__(env, strategy, gamma, **kwargs)

        self.alpha = alpha 

        # Initialize Q-table
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))

        # State to track epsilon decay
        self._current_episode = 0

    def get_parameters(self) -> dict:
        """
        Returns the parameters of the Q-Learning algorithm.
        """
        par = super().get_parameters()
        par["alpha"] = self.alpha
        return par

    def choose_action(self, state: int, episode) -> int: 

        # Use the helper function for action selection
        return self.strategy.action(self.q_table, state, episode)


    def update(self, state: int, action: int, 
               reward: float, next_state: int, 
               terminated: bool, truncated: bool):
         """
         Updates the Q-table based on the Q-Learning update rule.
         Called by the main training loop after each step.
         """
         # Q-Learning Update Rule: Q(s, a) = Q(s, a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s, a)]
         # Note: Q-Learning is off-policy, so we use the max Q value in the next state
         max_next_q = np.max(self.q_table[next_state])
         td_target = reward + self.gamma * max_next_q * (1 - terminated) # If terminated, gamma * max_next_q is 0
         td_error = td_target - self.q_table[state][action]
         self.q_table[state][action] = self.q_table[state][action] + self.alpha * td_error

         # If the episode ended, increment the episode counter for epsilon decay
         if terminated or truncated:
             self._current_episode += 1

    def get_intestines(self):
        return self.q_table

    def get_policy(self) -> np.ndarray:
        """
        Returns the greedy policy derived from the Q-table.
        """
        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
             policy[s] = np.argmax(self.q_table[s])
        return policy

    def get_value_function(self) -> np.ndarray:
        """
        Returns the value function derived from the Q-table (max Q for each state).
        """
        value_function = np.zeros(self.n_states)
        for s in range(self.n_states):
             value_function[s] = np.max(self.q_table[s])
        return value_function

    # ... save and load methods
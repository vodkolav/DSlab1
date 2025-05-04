# algorithms/q_learning.py
import gymnasium as gym
import numpy as np
from collections import defaultdict
import random
# Assuming you have a policies.py in utils
from utils.policies import epsilon_greedy_action
from algorithms.base_algorithm import RLAlgorithm

class QLearning(RLAlgorithm):
    """
    Q-Learning algorithm implementation.
    """
    def __init__(self, env: gym.Env, gamma: float = 1.0, alpha: float = 0.1,
                 initial_epsilon: float = 1.0, min_epsilon: float = 0.01,
                 epsilon_decay_episodes: int = 1, **kwargs):
        """
        Initializes the Q-Learning algorithm.

        Args:
            env: The Gymnasium environment.
            gamma: Discount factor.
            alpha: Learning rate.
            initial_epsilon: Starting value for epsilon in epsilon-greedy policy.
            min_epsilon: Minimum value for epsilon.
            epsilon_decay_episodes: The number of episodes over which epsilon decays from initial to min.
            **kwargs: Additional parameters for the base class.
        """
        super().__init__(env, gamma, **kwargs)

        self.alpha = alpha
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_episodes = epsilon_decay_episodes


        # Initialize Q-table
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))

        # State to track epsilon decay
        self._current_episode = 0


    def choose_action(self, state: int) -> int:
        """
        Selects an action using an epsilon-greedy policy.
        Epsilon decays over episodes managed by the main loop or internally.
        """
        # Calculate current epsilon based on episode count
        decay_rate = (self.initial_epsilon - self.min_epsilon) / self.epsilon_decay_episodes if self.epsilon_decay_episodes > 0 else 0
        epsilon = max(self.min_epsilon, self.initial_epsilon - decay_rate * self._current_episode)

        # Use the helper function for action selection
        return epsilon_greedy_action(self.q_table, state, epsilon, self.n_actions)


    def update(self, state: int, action: int, reward: float, next_state: int, terminated: bool, truncated: bool):
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
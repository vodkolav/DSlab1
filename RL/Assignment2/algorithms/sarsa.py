# algorithms/sarsa.py
import gymnasium as gym
import numpy as np
from collections import defaultdict
import random
from algorithms.base_algorithm import RLAlgorithm
from utils.strategy import Strategy

class SARSA(RLAlgorithm):

    def __init__(self,  env: gym.Env, strategy: Strategy,
                 gamma: float = 1.0, alpha: float = 0.1, **kwargs):
        """
        Performs SARSA to find the optimal action-value function.

        Args:
            env: The Gymnasium environment.
            alpha: Learning rate.
            gamma: Discount factor.
            initial_epsilon: Starting value for epsilon in epsilon-greedy policy.
            min_epsilon: Minimum value for epsilon.
            epsilon_decay_ratio: Decay rate for epsilon per episode.

        Returns:
            Q: The learned action-value function (defaultdict).
            policy: The learned epsilon-greedy policy (function).
        """

        super().__init__(env, strategy, gamma, **kwargs)

        self.alpha = alpha

        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))


    def choose_action(self, state: int, episode: int) -> int:
        # Choose the first action using the epsilon-greedy policy
        return self.strategy.action(self.q_table, state , episode)


    def choose_greedy_action(self, state: int) -> int:
        return self.strategy.greedy(self.q_table, state)


    def update(self, state: int, action: int, 
               reward: float, next_state: int, 
               terminated: bool, truncated: bool):
               
        # Choose the next action using the epsilon-greedy policy
        next_action = self.strategy.epsilon_greedy(self.q_table, state)

        # SARSA Update Rule: Q(s, a) = Q(s, a) + alpha * [reward + gamma * Q(s', a') - Q(s, a)]
        # Note: SARSA is on-policy, so we use the Q value of the *next action taken*
        td_target = reward + self.gamma * self.q_table[next_state][next_action] * (1 - terminated) # If terminated, gamma * Q(s',a') is 0
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] = self.q_table[state][action] + self.alpha * td_error

    def get_parameters(self) -> dict:
        """
        Returns the parameters of the Sarsa algorithm.
        """
        par = super().get_parameters()
        par["alpha"] = self.alpha
        return par
  
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


# algorithms/q_learning.py
import gymnasium as gym
import numpy as np
from collections import defaultdict
from utils.strategy import Strategy
from algorithms.agent import RLAgent


class TemporalDifference(RLAgent):
    """
    TemporalDifference algorithm implementation.
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
        self.V = defaultdict(lambda: np.zeros(1))


    def get_parameters(self) -> dict:
        """
        Returns the parameters of the Q-Learning algorithm.
        """
        par = super().get_parameters()
        par["alpha"] = self.alpha
        return par

    def choose_action(self, state: int, episode) -> int: 
        action = self.strategy.random() 
        #self.strategy.action(self.V, state, episode)

        return action

    def choose_greedy_action(self, state: int) -> int:
        return self.strategy.greedy(self.V, state)

    def update(self, state: int, action: int, 
               reward: float, next_state: int, 
               terminated: bool, truncated: bool):
        """
        Performs TD(0) Prediction to estimate the value function for a given policy.
        Called by the main training loop after each step.
        """
        # TD(0) Update Rule: V(s) = V(s) + alpha * [reward + gamma * V(s') - V(s)]

        td_target = reward + self.gamma * self.V[next_state][0] * (1 - terminated) # Gamma * V(next_state) is 0 if terminated
        td_error = td_target - self.V[state][0]
        self.V[state][0] = self.V[state][0] + self.alpha * td_error

    def get_intestines(self):
        return self.V

    def get_policy(self) -> np.ndarray:
        """
        Returns the greedy policy derived from the Q-table.
        """
        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
             policy[s] = np.argmax(self.V[s])
        return policy

    def get_value_function(self) -> np.ndarray:
        """
        Returns the value function derived from the Q-table (max Q for each state).
        """
        value_function = np.zeros(self.n_states)
        for s in range(self.n_states):
             value_function[s] = np.max(self.V[s])
        return value_function

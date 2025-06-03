# algorithms/sarsa.py
import gymnasium as gym
import numpy as np
from collections import defaultdict
from algorithms.agent import RLAgent
from algorithms.eligibility_traces import EligibilityTraces
from utils.strategy import Strategy

class SARSA(RLAgent):
    """
    SARSA algorithm implementation.
    """
    def __init__(self, env: gym.Env, strategy: Strategy,
                 gamma: float = 1.0, alpha: float = 0.1,
                 lambda_: float = 0.0,  **kwargs):
        """
        Initializes the SARSA algorithm.

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
        self.ET = EligibilityTraces(self, lambda_) 

    def choose_action(self, state: int, episode: int) -> int:
        return self.strategy.action(self.q_table, state, episode)


    def choose_greedy_action(self, state: int) -> int:
        return self.strategy.greedy(self.q_table, state)

    def size(self):
        res = super().default_dictionary_size(self.q_table) +\
            self.ET.size()
        return res 

    def update(self, state: int, action: int, 
               reward: float, next_state: int, 
               terminated: bool, truncated: bool):
        """
        Updates the Q-table based on the SARSA update rule.
        Called by the main training loop after each step.
        """
        # SARSA Update Rule: 
        # Q(s, a) = Q(s, a) + alpha * [reward + gamma * Q(s', a') - Q(s, a)]
        # Note: SARSA is on-policy, so we use the Q value of the *next action taken*
        # Choose the next action using the epsilon-greedy policy

        next_action = self.strategy.epsilon_greedy(self.q_table, state)
        td_target = reward + self.gamma * self.q_table[next_state][next_action] * (1 - terminated) # If terminated, gamma * Q(s',a') is 0
        td_error = td_target - self.q_table[state][action]
        
        if self.ET.enabled:            
            self.ET.update(state, action, td_error)
        else:
            # If no eligibility traces, just update the Q-table directly
            self.q_table[state][action] = self.q_table[state][action] + self.alpha * td_error


    def get_parameters(self) -> dict:
        """
        Returns the parameters of the algorithm.
        """
        par = super().get_parameters()
        par["alpha"] = self.alpha
        add = f" \\w E.T. (λ={self.ET.lambda_})" if self.ET.enabled else ""
        par["name"] = par["name"] + add
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


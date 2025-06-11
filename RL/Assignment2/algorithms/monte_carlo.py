
import gymnasium as gym
import numpy as np
from collections import defaultdict

from utils.strategy import Strategy
from algorithms.agent import RLAgent


class MonteCarlo(RLAgent):
    def __init__(self, env: gym.Env, strategy: Strategy,
                 gamma: float = 1.0, vizit:str = "every",
                    **kwargs):
        """
        Monte Carlo Control using GLIE (Every-Visit MC).

        Args:
            env: The Gymnasium environment.
            gamma: Discount factor.
            vizit: Type of visit: ["every"|"first"] Every-Visit MC or First-Visit MC

        Returns:
            Q: The learned action-value function (defaultdict).
            policy: The learned epsilon-greedy policy (function).
        """
        super().__init__(env, strategy, gamma, **kwargs)
        
        n_actions = env.action_space.n
        # Using defaultdict to handle states not yet visited
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.Returns_sum = defaultdict(lambda: np.zeros(n_actions))
        self.N_visits = defaultdict(lambda: np.zeros(n_actions))
        self.History = []
        self.vizit = vizit

    def get_parameters(self) -> dict:
        """
        Returns the parameters of the algorithm.
        """
        par = super().get_parameters()
        par["params"]["vizit"] = self.vizit
        return par

    def size(self):
        res = super().default_dictionary_size(self.q_table) +\
              super().default_dictionary_size(self.Returns_sum) +\
              super().default_dictionary_size(self.N_visits)+\
              super().default_dictionary_size(self.History)
        return res 
    
    def choose_action(self, state: int, episode) -> int: 
        action = self.strategy.action(self.q_table, state, episode)
        return action

    def choose_greedy_action(self, state: int) -> int:
        return self.strategy.greedy(self.q_table, state)

    def update(self, state: int, action: int, 
               reward: float, next_state: int, 
               terminated: bool, truncated: bool):        

        if not terminated and not truncated:
            # Log each step unless episode is over
            self.History.append((state, action, reward))

        else: 
            # Update Q-values after the episode
            G = 0 # Return
            # Iterate through the episode in reverse to calculate returns
            for t in range(len(self.History) - 1, -1, -1):
                state_t, action_t, reward_t = self.History[t]
                G = self.gamma * G + reward_t

                # Every-Visit MC: Update for every time a state-action pair is visited
                # For First-Visit MC, you would need to check if (state_t, action_t) has appeared before in the episode
                # from time t onwards.
                self.Returns_sum[state_t][action_t] += G
                self.N_visits[state_t][action_t] += 1
                self.q_table[state_t][action_t] = self.Returns_sum[state_t][action_t] \
                                             / self.N_visits[state_t][action_t]
            self.History = []

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

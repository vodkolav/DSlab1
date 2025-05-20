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

        #n_actions = env.action_space.n
        # Using defaultdict to handle states not yet visited
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))

        #epsilon = initial_epsilon

        # State to track epsilon decay
        #self._current_episode = 0
        #for i_episode in range(1, n_episodes + 1):

    def choose_action(self, state: int, episode: int) -> int:
       
      
        # Choose the first action using the epsilon-greedy policy
        return self.strategy.action(self.q_table, state , episode)

       
            # terminated = False
            # truncated = False

        #    while not terminated and not truncated:

    def update(self, state: int, action: int, 
               reward: float, next_state: int, 
               terminated: bool, truncated: bool):
       
        #next_state, reward, terminated, truncated, _ = env.step(action)
        
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




        # # The final policy is typically the greedy policy based on the learned Q-values
        # # Or you might return the epsilon-greedy policy if that's what you want to evaluate
        # final_policy = epsilon_greedy_policy(self.q_table, 0, n_actions) # Greedy policy

        # return self.q_table, final_policy


    # if __name__ == '__main__':
    #     # Example Usage:
    #     env = gym.make("Taxi-v3") # Use render_mode='human' to visualize

    #     print("Running SARSA...")
    #     learned_Q_sarsa, learned_policy_sarsa = sarsa(env, n_episodes=50000) # You'll need a good number of episodes
    #     print("SARSA Finished.")

    #     # You can test the learned policy
    #     print("\nTesting learned policy:")
    #     state, _ = env.reset()
    #     terminated = False
    #     truncated = False
    #     total_reward = 0
    #     while not terminated and not truncated:
    #         action = learned_policy_sarsa(state)
    #         state, reward, terminated, truncated, _ = env.step(action)
    #         total_reward += reward
    #         # Optional: env.render() if render_mode='human' is set


    #     print(f"Test episode finished with total reward: {total_reward}")


    #     env.close()
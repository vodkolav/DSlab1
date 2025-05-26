import gymnasium as gym
import numpy as np
from collections import defaultdict
from utils.strategy import Strategy
from algorithms.agent import RLAgent

class PolicyIteration(RLAgent):
    """
    Policy Iteration algorithm implementation.
    """

    def __init__(self, env: gym.Env, strategy: Strategy,
                 gamma: float = 1.0, theta: float = 1e-5,
                  **kwargs):
        """
        Initializes the Q-Learning algorithm.

        Args:
            env: The Gymnasium environment.
            gamma: Discount factor.
            theta: Convergence threshold.
            **kwargs: Additional parameters for the base class.
        """
        super().__init__(env, strategy, gamma, **kwargs)

        self.theta = theta 

        self.policy = np.random.randint(0, self.n_actions, self.n_states)
        self.V = defaultdict(lambda: np.zeros(1))


        # Initialize Q-table
        #self.q_table = defaultdict(lambda: np.zeros(self.n_actions))



    def choose_action(self, state: int, episode) -> int: 

        self.V = self.policy_evaluation(self.policy, self.V)
        return 0


    def choose_greedy_action(self, state: int) -> int:
        return self.policy[state]


    def update(self, state: int, action: int, 
               reward: float, next_state: int, 
               terminated: bool, truncated: bool):
        """
        Updates the Q-table based on the Q-Learning update rule.
        Called by the main training loop after each step.
        """
        new_policy, policy_stable = self.policy_improvement(self.V)
        self.policy = new_policy
        if policy_stable:
            self.terminate_prematurely = True


    def policy_evaluation(self, policy, V):
        """
        Evaluates the value function for a given policy.

        Args:
            env: The Gymnasium environment (must have a P attribute).
            policy: The policy to evaluate (numpy array mapping state to action).
            gamma: Discount factor.
            theta: Convergence threshold.

        Returns:
            V: The value function for the given policy (numpy array).
        """

        while True:
            delta = 0
            a = 0
            for s in range(self.n_states):
                v = V[s][a]
                new_v = 0
                action = policy[s]
                # Calculate the value of state s under the given policy
                # This part needs to compute sum_{s', r} P(s', r | s, policy[s]) * (r + gamma * V[s'])
                for prob, next_state, reward, terminated in self.env.unwrapped.P[s][action]:
                    new_v += prob * (reward + self.gamma * V[next_state][a])

                V[s][a] = new_v
                delta = max(delta, abs(v - V[s][a]))

            if delta < self.theta:
                #print(f"policy_evaluation, delta: {delta}")

                break
        return V

    def policy_improvement(self, V):
        """
        Improves the policy based on the current value function.

        Args:
            env: The Gymnasium environment (must have a P attribute).
            V: The current value function (numpy array).
            gamma: Discount factor.

        Returns:
            new_policy: The improved deterministic policy (numpy array).
            policy_stable: Boolean indicating if the policy changed.
        """
        new_policy = self.policy #np.zeros(self.n_states, dtype=int)
        policy_stable = True

        for s in range(self.n_states):
            old_action = new_policy[s]
            q_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for prob, next_state, reward, terminated in self.env.unwrapped.P[s][a]:
                    q_values[a] += prob * (reward + self.gamma * V[next_state][0])

            new_policy[s] = np.argmax(q_values)
            if old_action != new_policy[s]:
                policy_stable = False

        #print(f"policy_improvement" )

        return new_policy, policy_stable

    def get_parameters(self) -> dict:
        """
        Returns the parameters of the Q-Learning algorithm.
        """
        par = super().get_parameters()
        par["theta"] = self.theta
        return par

    def get_intestines(self):
        return self.V

    def get_policy(self) -> np.ndarray:
        """
        Returns the greedy policy derived from the Q-table.
        """
        return self.policy

    def get_value_function(self) -> np.ndarray:
        """
        Returns the value function derived from the Q-table (max Q for each state).
        """

        return self.V

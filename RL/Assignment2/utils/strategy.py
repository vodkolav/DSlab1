# utils/policies.py
import numpy as np
import random
from collections import defaultdict # Import defaultdict as the Q table might be one

class Strategy:

    def __init__(self, n_actions: int = 2,
                 decay: str = "linear",
                 initial_epsilon: float = 1.0, 
                 min_epsilon: float = 0.01,
                 epsilon_decay_episodes: int = 1):
        """_summary_

        Args:
            initial_epsilon: Starting value for epsilon in epsilon-greedy strategy.
            min_epsilon: Minimum value for epsilon.
            epsilon_decay_episodes: The number of episodes over which epsilon decays from initial to min.
        """
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.decay = decay 
        self.epsilon = initial_epsilon
        self.n_actions = n_actions

    def get_parameters(self) -> dict:
        """
        Returns the parameters of the strategy.
        """
        return {
            "name": "EpsilonGreedy",
            "params": {
                "initial_epsilon": self.initial_epsilon,
                "min_epsilon": self.min_epsilon,
                "epsilon_decay_episodes": self.epsilon_decay_episodes,
                "decay": self.decay
            }
        }

    def action(self, Q: defaultdict, 
               state: int, episode: int) -> int:
        if self.decay == "None":
            return self.epsilon_greedy(Q,state)
        elif self.decay == "linear":
            return self.epsilon_greedy_with_decay(Q, state, episode)
        else:
            # If state is neither, return random
            return self.random()

    def random(self):
        return random.randint(0, self.n_actions - 1)

    def greedy(self, Q: defaultdict, state: int) -> int:
        """
        Selects an action for a given state using greedy strategy: 
        Choose the action with the highest Q-value for the state

        Args:
            Q: The action-value function (a defaultdict or similar).
            state: The current state observation.

        Returns:
            The chosen action.
        """

        # Handle case where state might not be in Q if using defaultdict
        if state in Q and any(Q[state] != 0): # Check if state has non-zero Q values
            return np.argmax(Q[state])
        else:
            # If state is not in Q or all Q values are 0, explore
            return random.randint(0, self.n_actions - 1)


    def epsilon_greedy(self, Q: defaultdict, 
                             state: int) -> int:
        """
        Selects an action for a given state using an epsilon-greedy strategy.

        Args:
            Q: The action-value function (a defaultdict or similar).
            state: The current state observation.
            epsilon: The probability of choosing a random action.
            n_actions: The total number of possible actions.

        Returns:
            The chosen action.
        """
        if random.random() < self.epsilon:
            # Explore: Choose a random action
            return self.random()
        else:
            # Exploit: Choose the action with the highest Q-value for the state
            return self.greedy(Q, state)
            

    def epsilon_greedy_with_decay(self, Q: defaultdict, 
                                  state: int,  episode ) -> int:
        """
            Epsilon decays over episodes managed by the main loop or internally.
        """
        # Calculate current epsilon based on episode count
        decay_rate = (self.initial_epsilon - self.min_epsilon) / \
                            self.epsilon_decay_episodes if self.epsilon_decay_episodes > 0 else 0
        self.epsilon = max(self.min_epsilon, self.initial_epsilon - decay_rate * episode)

        return self.epsilon_greedy(Q, state)


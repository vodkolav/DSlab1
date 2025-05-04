# utils/policies.py
import numpy as np
import random
from collections import defaultdict # Import defaultdict as the Q table might be one

def epsilon_greedy_action(Q: defaultdict, state: int, epsilon: float, n_actions: int) -> int:
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
    if random.random() < epsilon:
        # Explore: Choose a random action
        return random.randint(0, n_actions - 1)
    else:
        # Exploit: Choose the action with the highest Q-value for the state
        # Handle case where state might not be in Q if using defaultdict
        if state in Q and any(Q[state] != 0): # Check if state has non-zero Q values
             return np.argmax(Q[state])
        else:
             # If state is not in Q or all Q values are 0, explore
             return random.randint(0, n_actions - 1)
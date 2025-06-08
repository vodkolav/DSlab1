from algorithms.agent import RLAgent
import numpy as np
from collections import defaultdict


class EligibilityTraces:
    def __init__(self, host: RLAgent , lambda_=0.0):
        """A plugin for RLAgent that implements eligibility traces.
        if lambda_ is set to 0, it will not use eligibility traces.
        
        Args:
            host (RLAgent): the RLAgent that this plugin will be attached to.
            lambda_ (float, optional): lambda. Defaults to 0.0 (by default, no eligibility traces are used).

        Raises:
            ValueError: If lambda_ is not between 0 and 1.
        """
        self.host = host
        self.lambda_ = lambda_
        
        # Initialize eligibility traces for the episode
        if lambda_ > 0.0:
            if not (0 <= lambda_ <= 1):
                raise ValueError("lambda_ must be between 0 and 1.")
            # If lambda_ is specified, we use eligibility traces
            self.lambda_ = lambda_
            # Eligibility traces are initialized to zero for each state-action pair
            # This will be updated during the episode
            self.E = defaultdict(lambda: np.zeros(host.n_actions))
            self.enabled = True
        else:
            # If lambda_ is 0, we do not use eligibility traces
            self.E = None
            self.enabled = False

    def size(self):
        if self.enabled:
            return self.host.default_dictionary_size(self.E) 
        else:
            return 0

    def moniker(self): 
        return f" \\w E.T. (λ={self.lambda_})" if self.enabled else ""


    def get_parameters(self) -> dict:
        """
        Returns the parameters of the eligibility traces.
        """
        par = {
            "lambda_": self.lambda_
        }
        return par


    def reset(self):
        self.E = defaultdict(lambda: np.zeros(self.host.n_actions))

    def update(self, state, action, td_error: float, terminated: bool = False):

        # Update eligibility trace for the current state - action pair
        # For state-value prediction, the trace for state s is typically incremented by 1

        self.E[state][action] += 1
        # Update value function and decay eligibility traces
        for s in self.host.q_table: # Iterate through all states that have been visited
            for a in range(self.host.n_actions): # Iterate through all actions that have been visited
                if self.E[s][a] > 0: # Only update states with non-zero trace
                    self.host.q_table[s][a] = self.host.q_table[s][a] + self.host.alpha * td_error * self.E[s][a]
                    # Decay trace
                    self.E[s][a] = self.host.gamma * self.lambda_ * self.E[s][a]
        if terminated:
            # If the episode is terminated, reset the eligibility traces
            self.reset()

    def get_trace(self):
        return self.E
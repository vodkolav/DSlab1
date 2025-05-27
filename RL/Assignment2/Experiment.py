from algorithms.agent import RLAgent
from telemetry.telemetry_manager import TelemetryManager


import gymnasium as gym
import numpy as np


class Experiment:
    """
    Class to manage the RL experiment, including training and evaluation.

        Args:
            env: The Gymnasium environment.
            algorithm: The RLAlgorithm instance.
    """
    def __init__(self, env: gym.Env, algorithm: RLAgent):
        self.env = env
        self.algorithm = algorithm
        # Create Telemetry Manager
        self.telemetry = TelemetryManager(env, algorithm)


    def run_episode(self, i_episode):
        """
        Runs a single episode in the environment.

            Args:
                i_episode: episode number

            Returns:
                The total reward for the episode.
        """
        state, info = self.env.reset()
        terminated = False
        truncated = False
        total_reward = 0

        if self.telemetry and self.is_training:
            self.telemetry.reset_episode_metrics()

        if self.render:
            frame = self.env.render()

        while not terminated and not truncated:
            # Algorithm chooses an action
            # Note: For evaluation, choose_action should ideally be purely greedy.
            # The QLearning class handles this internally based on episode count for epsilon decay.
            # For evaluation phase, ensure epsilon is effectively 0.
            if self.is_training:
                action = self.algorithm.choose_action(state, i_episode) # choose_action now handles exploration strategy
            else: 
                action = self.algorithm.choose_greedy_action(state)

            if action is None:
                print("wtf")

            # Environment takes a step
            next_state, reward, terminated, truncated, info = self.env.step(action)

            # Algorithm updates its internal state if training
            if self.is_training:
                # Pass all relevant info to the update method
                self.algorithm.update(state, action, reward, next_state, terminated, truncated) # Update signature might vary

            total_reward += reward
            state = next_state # Move to the next state
                
            if self.telemetry:
                #if self.render:
                frame = self.env.render()    

                self.telemetry.record_step(reward, info, frame)


        if self.telemetry:
            self.telemetry.record_episode_end(self.algorithm, self.is_training ) # Record episode number

        return self.algorithm.terminate_prematurely


    def train_algorithm(self, num_episodes: int):
        """
        Runs the training loop for an episode-based algorithm.

        Args:
            num_episodes: Total number of episodes for training.
        """
        self.is_training = True # Set to True for training
        self.telemetry.report_start(self.is_training, num_episodes) # Start telemetry reporting
        for i_episode in range(num_episodes):

            self.run_episode(i_episode) # No rendering during training usually
            if self.algorithm.terminate_prematurely:
                print("Algorithm decided to terminate prematurely.")
                break
            self.telemetry.report_progress()    

        self.telemetry.report_end() # End telemetry reporting


    def evaluate_algorithm(self, num_episodes: int = 10):
        """
        Evaluates the learned policy of an algorithm.

        Args:
            num_episodes: Number of evaluation episodes.
        """
        self.is_training = False # Set to False for evaluation
        
        self.telemetry.report_start(self.is_training, num_episodes)

        for i_episode in range(num_episodes):
            self.run_episode(i_episode)
            self.telemetry.report_progress()

        self.telemetry.report_end()


    # --- Special Handling for Dynamic Programming ---
    def run_dynamic_programming(algorithm):
        """
        Runs a Dynamic Programming algorithm (which doesn't use step-by-step interaction loops in main).
        """
        print(f"--- Running Dynamic Programming: {type(algorithm).__name__} ---")
        # DP algorithms typically have a 'solve' or 'run' method that computes the policy/value function once
        algorithm.solve() # Assuming DP class has a 'solve' method

        print("Dynamic Programming finished.")
        # You would then get the policy/value function using algorithm.get_policy() or algorithm.get_value_function()

    def run_experiment(self,train_episodes, eval_episodes, render = False):

        self.render = render

        self.telemetry.total_episodes = train_episodes + eval_episodes

        self.train_algorithm(train_episodes)

        # --- Optional: Evaluate the trained algorithm ---
        self.evaluate_algorithm(eval_episodes)


        # --- Example for Dynamic Programming (Different Structure) ---
        # DP algorithms don't fit the step-by-step update in the same training loop
        # dp_agent = DynamicProgramming(env, gamma=0.99, theta=1e-5)
        # run_dynamic_programming(dp_agent)
        # After running DP, you would evaluate its *policy* directly using run_episode with is_training=False

        # Close the environment
        self.env.close()
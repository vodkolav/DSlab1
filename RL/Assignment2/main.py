
# main.py
import gymnasium as gym
# Import the algorithm(s) you want to run
from algorithms.q_learning import QLearning
from algorithms.base_algorithm import RLAlgorithm
# from algorithms.sarsa import SARSA
# from algorithms.dynamic_programming import DynamicProgramming # DP will be handled differently

from telemetry.telemetry_manager import TelemetryManager
import numpy as np

class Experiment:
    """
    Class to manage the RL experiment, including training and evaluation.
    """
    def __init__(self, env: gym.Env, algorithm: RLAlgorithm):
        self.env = env
        self.algorithm = algorithm
        # Create Telemetry Manager
        self.telemetry = TelemetryManager()


    def run_episode(self):
        """
        Runs a single episode in the environment.

        Args:
            env: The Gymnasium environment.
            algorithm: The RLAlgorithm instance.
            is_training: True if in training mode (call algorithm.update), False for evaluation.
            telemetry: Optional TelemetryManager for recording metrics.
            render: Whether to render the environment.

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
            print(self.env.render())

        while not terminated and not truncated:
            # Algorithm chooses an action
            # Note: For evaluation, choose_action should ideally be purely greedy.
            # The QLearning class handles this internally based on episode count for epsilon decay.
            # For evaluation phase, ensure epsilon is effectively 0.
            action = self.algorithm.choose_action(state) # choose_action now handles exploration strategy

            # Environment takes a step
            next_state, reward, terminated, truncated, info = self.env.step(action)

            # Algorithm updates its internal state if training
            if self.is_training:
                # Pass all relevant info to the update method
                self.algorithm.update(state, action, reward, next_state, terminated, truncated) # Update signature might vary

                if self.telemetry:
                    self.telemetry.record_step(reward, info)


            total_reward += reward
            state = next_state # Move to the next state

            if self.render:
                print(self.env.render())

        if self.telemetry and self.is_training:
            self.telemetry.record_episode_end(self.telemetry.get_total_episodes() + 1) # Record episode number

        return total_reward


    def train_algorithm(self, num_episodes: int):
        """
        Runs the training loop for an episode-based algorithm.

        Args:
            algorithm: The RLAlgorithm instance to train.
            env: The Gymnasium environment.
            total_episodes: Total number of episodes for training.
            telemetry: Optional TelemetryManager for recording metrics.
        """
        self.is_training = True # Set to True for training
        print(f"--- Starting Training for {type(self.algorithm).__name__} for {num_episodes} episodes ---")
        for i_episode in range(num_episodes):
            self.run_episode() # No rendering during training usually

            # Optional: Print progress
            if (i_episode + 1) % 1000 == 0:
                avg_reward = self.telemetry.get_average_reward() if self.telemetry else "N/A"
                print(f"Episode {i_episode + 1}/{num_episodes}, Avg Reward (last 100): {avg_reward:.2f}")

        print("Training finished.")


    def evaluate_algorithm(self, num_episodes: int = 10):
        """
        Evaluates the learned policy of an algorithm.

        Args:
            algorithm: The RLAlgorithm instance to evaluate.
            env: The Gymnasium environment.
            num_episodes: Number of evaluation episodes.
            render: Whether to render the environment during evaluation.
        """
        self.is_training = False # Set to False for evaluation
        print(f"\n--- Running Evaluation for {type(self.algorithm).__name__} over {num_episodes} episodes ---")
        
        episode_rewards = []
        # Temporarily set epsilon to 0 for greedy evaluation if the algorithm uses it internally
        original_epsilon = getattr(self.algorithm, '_current_epsilon', None)
        if original_epsilon is not None:
            setattr(self.algorithm, '_current_epsilon', 0.0) # Force greedy during evaluation


        for i_episode in range(num_episodes):
            total_reward = self.run_episode()
            episode_rewards.append(total_reward)
            print(f"  Evaluation Episode {i_episode + 1}: Total Reward = {total_reward}")

        # Restore original epsilon after evaluation
        if original_epsilon is not None:
            setattr(self.algorithm, '_current_epsilon', original_epsilon)

        avg_reward = np.mean(episode_rewards)
        print(f"Average Evaluation Reward over {num_episodes} episodes: {avg_reward:.2f}")
        print("Evaluation finished.")


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



if __name__ == '__main__':
    # Configuration
    ENV_ID = "Taxi-v3"
    NUM_TRAINING_EPISODES = 2 #50000
    NUM_EVAL_EPISODES = 2 #10
    RENDER_EVALUATION = True

    # --- Instantiate and Train an Algorithm ---
    # Example: Q-Learning
    q_learning_params = {
        "gamma": 0.99,
        "alpha": 0.1,
        "initial_epsilon": 1.0,
        "min_epsilon": 0.01,
        "epsilon_decay_episodes": NUM_TRAINING_EPISODES / 2 # Decay epsilon over half the training episodes
    }

    # Create environment
    # Use render_mode='human' for rendering during evaluation if RENDER_EVALUATION is True
    # 
    env = gym.make(ENV_ID, render_mode='ansi' if RENDER_EVALUATION else None)

    q_learning_agent = QLearning(env, **q_learning_params)

    exper = Experiment(env, q_learning_agent)
    exper.run_experiment(NUM_TRAINING_EPISODES, NUM_EVAL_EPISODES ,RENDER_EVALUATION)

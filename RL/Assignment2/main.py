
# main.py
import gymnasium as gym
# Import the algorithm(s) you want to run
from RL.Assignment2.Experiment import Experiment
from algorithms.q_learning import QLearning
# from algorithms.sarsa import SARSA
# from algorithms.dynamic_programming import DynamicProgramming # DP will be handled differently

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

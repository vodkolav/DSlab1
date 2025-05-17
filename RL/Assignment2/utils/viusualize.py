import matplotlib.pyplot as plt
import numpy as np

def visualize_training_progress(q_learning_agent, telemetry_manager):
    # --- Visualize Training Metrics (Example: Episode Rewards) ---
    if telemetry_manager.episode_rewards:
        plt.figure(figsize=(12, 6))
        plt.plot(telemetry_manager.episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"Training Progress for {type(q_learning_agent).__name__}")
        plt.grid(True)
        plt.show()

        # Optional: Plot moving average
        window_size = 100
        if len(telemetry_manager.episode_rewards) >= window_size:
             moving_avg = np.convolve(telemetry_manager.episode_rewards, np.ones(window_size)/window_size, mode='valid')
             plt.figure(figsize=(12, 6))
             plt.plot(moving_avg)
             plt.xlabel(f"Episode (Smoothed over {window_size} episodes)")
             plt.ylabel(f"Average Total Reward (last {window_size} episodes)")
             plt.title(f"Smoothed Training Progress for {type(q_learning_agent).__name__}")
             plt.grid(True)
             plt.show()

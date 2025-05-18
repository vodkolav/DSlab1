import matplotlib.pyplot as plt
import numpy as np

import time

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




import time
from IPython.display import display, clear_output

def play_ansi_episode(frames: list[str], interval: float = 0.1):
    """
    Plays a sequence of ANSI frames as an animation in a Jupyter cell.

    Args:
        frames: A list of strings, where each string is an ANSI frame.
        interval: The time in seconds to pause between frames.
    """
    if not frames:
        print("No frames to display.")
        return

    print(f"Playing animation with {len(frames)} frames (interval: {interval}s)...")

    try:
        for i, frame in enumerate(frames):
            clear_output(wait=True) # Clear the previous frame, wait for the next print
            print(frame)          # Display the current frame

            # Add a frame counter (optional)
            print(f"Frame: {i + 1}/{len(frames)}")

            if i < len(frames) - 1:
                time.sleep(interval) # Pause between frames

        # Keep the last frame visible
        print("\nAnimation finished.")

    except KeyboardInterrupt:
        print("\nAnimation stopped.")
        # Keep the current frame visible if stopped manually
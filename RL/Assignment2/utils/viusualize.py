import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pandasql as ps

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


def summarize_Q(episode, actions):
    Q_t = pd.DataFrame(episode["internal_state"]).T
    Q_t.columns = actions.split(",")
    Q_t['state'] = Q_t.index.array
    Q_t['episode'] = episode.i
    Q_t['is_training'] = episode.is_training
    return Q_t


def TaxiObservationSpace():
    observation_space = {}

    for taxi_row in range(5):
        for taxi_col in range(5):
            for passenger_location in range(5):
                for destination in range(4):
                    state_num = ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
                    
                    state = {"State_Num": state_num,
                            "Taxi_Row": taxi_row, 
                            "Taxi_Col": taxi_col, 
                            "Passenger_Location": passenger_location, 
                            "Destination": destination}

                    observation_space[state_num] = state


    states_map = pd.DataFrame(observation_space).T
    return states_map


def Q_evolution(episodes_to_plot):
    actions = "down,up,right,left,pickup,dropoff" # order of actions is important!
    
    Qs = [summarize_Q(ep, actions) for i, ep in episodes_to_plot.iterrows()]
    
    Qs = pd.concat(Qs)
        
    states_map = TaxiObservationSpace()
    eps = episodes_to_plot[["i","is_training"]]  
    # build "timespace": all combinations of places and times in the world 
    timespace = eps.merge(states_map, how='cross')    

    # paint experiment data onto the timespace
    q = f"""
        SELECT ts.i as episode, ts.is_training, State_Num,
        Taxi_Row, Taxi_Col, Passenger_Location, Destination, 
        {actions}
    
        FROM timespace ts
        LEFT JOIN Qs as q ON ts.State_Num = q.state
                          AND ts.i = q.episode
                          AND ts.is_training = q.is_training      
        """    
    experiment_data = ps.sqldf(q).fillna(0)
    
    dirmap = {"down":"↓","up":"↑","right":"→","left":"←"}  
    experiment_data["pref_dir"]= experiment_data[["down","up","right","left"]].idxmax(axis="columns").map(dirmap)
    
    return experiment_data


def plot_Q_evolution(experiment_data,  metric = "pickup"):
    #metric = "dropoff"  
    actions = "down,up,right,left,pickup,dropoff" # order of actions is important!
    
    if metric == "pickup":
        grouping = ""
        goal = "Passenger_Location"
        condition = "Passenger_Location < 4"
    elif metric == "dropoff":
        grouping = ", Destination"
        goal = "Destination"
        condition = "Passenger_Location = 4" 
    else:
        raise ValueError("pickup| dropoff are the only valid metrics") 
    #print("metric: ", metric ,"|location: ", goal, "|condition: ",condition)
    # , count({a}) as {a}_cnt
    acts = [ f"min({a}) as {a}" for a in actions.split(",")]
    acts = "\n,".join(acts)
    
    q = f"""
        SELECT episode, is_training, 
        Taxi_Row, Taxi_Col, Passenger_Location {grouping} ,
        {acts}, pref_dir
    
        FROM experiment_data
        
        WHERE is_training = 1 and {condition}
        GROUP BY episode, is_training, 
                 Taxi_Row, Taxi_Col, Passenger_Location {grouping}
        ORDER BY Destination, episode 
        """
    #print(q)
    toplot = ps.sqldf(q)
        
    fig = px.scatter(toplot, x="Taxi_Col", y="Taxi_Row", 
                     #hover_data=[metric], 
                     text="pref_dir",
                     color=metric,  
                       #facet_row= "ExperimentId" , 
                       facet_col=goal,
                       color_continuous_scale="YlOrRd",
                       title=f"Q-Values for {metric} Action, Current goal: {goal}",
                       #width=400, #
                       height=500,
                       animation_frame="episode",
                       labels={"Taxi_Col": "Taxi Column", 
                               "Taxi_Row": "Taxi Row", 
                               "pickup": "Q-Value"}
                      )  
    
    fig.update_yaxes(autorange="reversed")
    
    fig.update_traces(textfont=dict(size=20, color="blue"),
                      marker=dict(size=20 ))
    return fig


def experiment_timeline(experiments, episodes, sort_param = "gamma", win_len = 50):
    experiments_smol =  experiments.copy() # .drop("strategy", axis=1)
    episodes_smol =  episodes.drop(["internal_state","replay", "info"], axis=1)
    
    q = f"""
    SELECT ex.id, ex.name, ex.moniker, ex.init_rss_mb as init_mem,  
    ex.alpha, ex.gamma, ex.theta, ex.decay, ex.lambda, ex.initial_epsilon, -- ex.vizit, 
    i, is_training, start, end, end - start as duration, 
    length, ep.reward, ep.rss_mb, ep.size_bytes, ep.epsilon, 

    avg(reward) OVER (
        PARTITION BY {sort_param}, is_training, exp_id
        ORDER BY i
        RANGE BETWEEN {win_len} PRECEDING AND CURRENT ROW
    ) AS avg_reward, 

    sum(reward) OVER (
        PARTITION BY {sort_param}, is_training, exp_id
        ORDER BY i
        RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumul_reward, 

    avg(length) OVER (
        PARTITION BY {sort_param}, is_training, exp_id
        ORDER BY i
        RANGE BETWEEN {win_len} PRECEDING AND CURRENT ROW
    ) AS avg_length

    FROM experiments_smol ex
    LEFT JOIN episodes_smol as ep ON ex.id = ep.exp_id
    -- WHERE is_training = 1
    ORDER BY moniker, {sort_param}, i
    """
    return ps.sqldf(q).fillna(0)
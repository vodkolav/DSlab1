# experiment/run_single_experiment.py

import gymnasium as gym
import os
from datetime import datetime
import json

import multiprocessing
import os

from pathlib import Path
import pandas as pd 


# Import your classes

from utils.strategy import Strategy # Assuming Strategy is a class you defined for policies
from Experiment import Experiment # Your Experiment class


from algorithms.dynamic_programming import PolicyIteration
from algorithms.monte_carlo import MonteCarlo
from algorithms.temporal_difference import TemporalDifference
from algorithms.q_learning import QLearning
from algorithms.sarsa import SARSA 


# --- NEW: Mapping of algorithm names to their classes ---
ALGORITHM_CLASSES = {
    "DynamicProgramming": PolicyIteration, # DP is handled differently, might not fit here
    "SARSA": SARSA,
    "QLearning": QLearning,
    "MonteCarlo": MonteCarlo,
    "TemporalDifference": TemporalDifference,
}


def run_case(exp_config: dict, output_dir = "data/results") -> dict:
    """
    Runs a single RL experiment based on the provided configuration.
    This function will be run by a separate process.

    Args:
        exp_config: A dictionary containing all parameters for this experiment.

    Returns:
        A dictionary containing key results and path to saved telemetry.
    """
    # exp_name = exp_config.get("name", "unnamed_experiment")
    # print(f"[{os.getpid()}] Starting experiment: {exp_name}")

    # --- Extract parameters ---
    env_id = exp_config["env"]["name"]

    meta = exp_config["metadata"]
    num_training_episodes = meta["num_training_episodes"]
    num_eval_episodes = meta["num_eval_episodes"]
    render_evaluation = meta.get("render_evaluation", False) # Don't render in parallel usually
    save_ansi_frames = meta.get("save_ansi_frames", False) # Or handle differently
    

    algorithm_name = exp_config["algorithm"]["name"]
    algo_params = exp_config["algorithm"]["params"]
    strategy_params = exp_config["strategy"]["params"]

    # --- Setup Environment ---
    # Environments are NOT picklable, so each process must create its own.
    # Set render_mode to None or 'ansi' as 'human' rendering often conflicts in parallel processes.
    render_mode = 'ansi' if save_ansi_frames else None
    env = gym.make(env_id, render_mode=render_mode)

    # --- Instantiate Strategy, Agent, Experiment ---
    strat = Strategy(env.action_space.n, **strategy_params)


        # Get the algorithm class from the mapping
    algorithm_class = ALGORITHM_CLASSES.get(algorithm_name)
    if algorithm_class is None:
        raise ValueError(f"Unknown algorithm name: {algorithm_name}. "
                         f"Available algorithms: {list(ALGORITHM_CLASSES.keys())}")

    agent = algorithm_class(env, strat, **algo_params) # Adjust based on the actual algorithm class
    exper = Experiment(env, agent, exp_config)

    # --- Run Experiment ---
    # The run_experiment method from your Experiment class
    # You might want to return a summary directly or save it.
    # For parallel runs, it's best to save telemetry to a unique file.    
    
    # Assuming your Experiment.run_experiment takes a telemetry object and potentially an output path
    exper.run_experiment(
        train_episodes=num_training_episodes,
        eval_episodes=num_eval_episodes,
        render=render_evaluation, # Likely False for parallel runs
    )

    # Save configuration and results specific to this run
    # config_filepath = os.path.join(output_dir, "config.json")
    # with open(config_filepath, 'w') as f:
    #     json.dump(exp_config, f, indent=4)


    run_timestamp = exper.telemetry.metadata["start_time"] 
    ex_id = exper.telemetry.metadata["id"]

    fname =  f"{ex_id}.json"
    os.makedirs(output_dir, exist_ok=True)
    telemetry_filepath = os.path.join(output_dir, fname)
    exper.telemetry.save_metrics(telemetry_filepath) # Save the telemetry

    env.close()

    result = {"status": "done", "experiment_id": str(ex_id), "timestamp":run_timestamp, "telemetry_filepath": telemetry_filepath}

    return result


def read_configs(config_filepath):
    try:
        with open(config_filepath, 'r') as f:
            experiment_configs = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_filepath}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {config_filepath}")
        return

    print(f"Loaded {len(experiment_configs)} experiments from {config_filepath}")
    return experiment_configs



def run_battery_of_experiments(experiment_configs: list, num_cores: int = None, results_dir="results"):
    """
    Reads experiment configurations from a JSON file and runs them in parallel.

    Args:
        config_filepath: Path to the JSON file containing experiment configurations.
        num_cores: Number of CPU cores to use. Defaults to all available cores.
    """
    if num_cores is None:
        num_cores = os.cpu_count()
        if num_cores is None:
            print("Warning: Could not detect CPU count, defaulting to 1 core.")
            num_cores = 1
        else:
            print(f"Detected {num_cores} CPU cores. Using {num_cores} workers.")

    # Separate every run of battery of tests to its own dir
    results_dir = results_dir + "/" + datetime.now().strftime("%Y%m%d-%H%M")
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a multiprocessing Pool
    # The 'with' statement ensures the pool is properly closed
    all_results = []
    with multiprocessing.Pool(processes=num_cores) as pool:
        # pool.apply_async submits a single task and returns an AsyncResult object immediately.
        # This allows you to submit all tasks without waiting for each one to finish.
        async_results = []
        for i, config in enumerate(experiment_configs):
            print(f"Submitting experiment {i+1}/{len(experiment_configs)}: {config.get('name', 'unnamed')}")
            result = pool.apply_async(run_case, (config,results_dir))
            async_results.append(result)

        # Wait for all tasks to complete and collect results
        print("\nWaiting for experiments to complete...")
        for i, res in enumerate(async_results):
            try:
                # .get() will block until the result is ready
                # You can add a timeout if you want to handle unresponsive processes
                experiment_result = res.get()
                all_results.append(experiment_result)
                print(f"Experiment {i+1}/{len(experiment_configs)}")
            except Exception as e:
                print(f"Error running experiment {i+1}: {e}")
                all_results.append({"error": str(e), "config": experiment_configs[i]})

    print("\nAll experiments finished.")
    print("\n--- Summary of Results ---")
    for res in all_results:
        if "error" in res:
            print(f"  FAILED: {res['config'].get('name', 'Unnamed')} - Error: {res['error']}")
        else:
            print(res["status"], res["timestamp"])

    return all_results, results_dir


def load_experiment(data):
    meta = data["metadata"]

    # Flatten the algorithm parameters into the metadata
    # I'll deal with strategy parameters later
    algo = data["algorithm"]
    algo.update(algo["params"])
    algo.pop("params", None)
    meta.update(algo)

    episodes = pd.DataFrame(data["episodes"])
    episodes["exp_id"] = meta["id"]

    return meta, episodes


def load_results(results_dir, patt = "*"):
    pth = Path(results_dir)
    print(pth.absolute())
    paths = list(pth.glob(patt +".json"))
    print(paths)
    experiments = ['']*len(paths)
    episodes = []
    for i,p in enumerate(paths):
        # try:
            with open(p) as f:
                data = json.load(f)
                experiment, exp_episodes = load_experiment(data)
                experiments[i] = experiment
                episodes.append(exp_episodes)
        # except Exception as ex: 
        #     print("oops:", p)
    experiments = pd.DataFrame(experiments)
    episodes = pd.concat(episodes)
    return experiments, episodes

if __name__ == '__main__':
    # It's crucial that code run by multiprocessing.Pool is either in another file
    # or inside the 'if __name__ == "__main__":' block itself.
    # The function run_single_experiment is imported, so it's safe.

    config_file = "configs/sarsa_battery.json"
    # Set num_cores to None to use all available CPU cores,
    # or a specific number like 4, 8, etc.
    results = run_battery_of_experiments(config_file, num_cores=None)

    # You can now further process or analyze 'results'
    # e.g., generate plots comparing performance, save summary to CSV, etc.
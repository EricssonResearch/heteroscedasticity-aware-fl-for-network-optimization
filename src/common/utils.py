import os
import socket
import time
from typing import List, Tuple
import json
from datetime import datetime

import numpy as np
from flwr.common import Metrics
import torch


# Config

def get_config_env_var(param_name, dtype=None):
    full_key = f"FAIRFL_CONFIG_{param_name.upper()}"
    value = os.environ.get(full_key)

    if value is not None and dtype is not None:
        value = dtype(value)

    return value


def set_config_env_var(param_name, value):
    full_key = f"FAIRFL_CONFIG_{param_name.upper()}"
    os.environ[full_key] = str(value)


def config_from_env_list(env_vars_list):
    env_vars = {}
    prefix = "FAIRFL_CONFIG_"

    for var, dtype in env_vars_list:
        value = os.environ.get(f"{prefix}{var.upper()}")
        if value is not None:
            env_vars[var] = dtype(value)

    return env_vars


def all_config_from_env():
    config_params = {}
    prefix = "FAIRFL_CONFIG_"

    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            config_params[config_key] = value

    return config_params


def non_zero_weights(weights):
    return weights if torch.any(weights) else torch.ones(len(weights))

def normalize_weights(weights):
    weights = non_zero_weights(weights)
    return weights/torch.sum(weights)


# Recollection priority

def uniform_recollection(recollection_size):
    def func(memorybank):
        return torch.ones(len(memorybank))
    return func

def var_reciprocal_recollection(recollection_size):
    def func(memorybank):
        weights = var_reciprocal(memorybank.noise_var_history).repeat_interleave(memorybank.n_checkpoint_samples)
        return non_zero_weights(weights)
    return func

def rolling_window_recollection(recollection_size):
    def func(memorybank):
        weights = torch.zeros(len(memorybank))
        weights[-recollection_size:] = 1
        return non_zero_weights(weights)
    return func

recollection_strategies = {
    "uniform": uniform_recollection,
    "reciprocal": var_reciprocal_recollection,
    "rolling_window": rolling_window_recollection
}

def parse_recollection_configs(config):
    if not config in recollection_strategies.keys():
        raise ValueError(f"Unexpected recollection strategy: {config}. Must be one of {list(recollection_strategies.keys())}")
    return config

def var_reciprocal(vars, eps=0.1):
    return 1/(vars + eps)


# Recollection size

def load_linear_recollection_size(default_size):
    return lambda memorybank: (1 - memorybank.load_history[-1].item()) * default_size

def constant_recollection_size(default_size):
    return default_size

def complete_recollection_size(default_size):
    return lambda memorybank: len(memorybank)

recollection_size_strategies = {
    "constant": constant_recollection_size,
    "linear": load_linear_recollection_size,
    "complete": complete_recollection_size,
}

def parse_recollection_size_configs(config):
    if not config in recollection_size_strategies.keys():
        raise ValueError(f"Unexpected recollection size strategy: {config}. Must be one of {list(recollection_size_strategies.keys())}")
    return config


# Load dynamics

def configure_recollection(recollection_config, recollection_size_config, default_recollection_size):
    recollection_strategy_name = parse_recollection_configs(recollection_config)
    recollection_size_strategy_name = parse_recollection_size_configs(recollection_size_config)
    recollection_strategy = recollection_strategies[recollection_strategy_name](default_recollection_size)
    recollection_size_strategy = recollection_size_strategies[recollection_size_strategy_name](default_recollection_size)
    return recollection_strategy, recollection_size_strategy


# Results

def results_from_history(history):
    results = {
        'losses_distributed': history.losses_distributed,
        'losses_centralized': history.losses_centralized,
        'metrics_distributed': history.metrics_distributed,
        'metrics_centralized': history.metrics_centralized
    }
    return results

def describe_simulation(sim_dict):
    return {
        "n_checkpoints": len(sim_dict["data"]["0"]["train"]["x"]),
        "n_checkpoint_train_samples": len(sim_dict["data"]["0"]["train"]["x"][0]),
        "n_checkpoint_val_samples": len(sim_dict["data"]["0"]["val"]["x"]),
        "n_dimensions": len(sim_dict["data"]["0"]["train"]["x"][0][0]),
        "load_schedules": sim_dict["load_schedules"]
    }


def save_experiment(history, sim_data):
    params = all_config_from_env()

    sim_info = describe_simulation(sim_data)
    results = results_from_history(history)

    experiment = {
        'date': datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        'params': params,
        'simulation': sim_info,
        'results': results
    }

    file_name = 'experiment-' + datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    # Check if "out" parameter is provided and not empty
    if params.get("out"):
        output_dir = params["out"]
    else:
        base_dir = os.path.dirname(params["sim"])
        output_dir = base_dir.replace('simulation', 'out')
        
        if params["algorithm"] == 'fedhaf':
            if float(params["tilt"]) == 0:
                algorithm_name = "fedavg"
            elif int(params["abs_fair"]) == 1:
                algorithm_name = "term_" + params["tilt"]
            else:
                algorithm_name = "fedhaf_" + params["tilt"]
        else:
            algorithm_name = params["algorithm"]

        output_dir = os.path.join(output_dir, algorithm_name)

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name + ".json")

    with open(file_path, 'w') as f:
        json.dump(experiment, f)


# Evaluation aggregation

def mse_alleatoric_var(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    client_ids, mses, noise_vars, clean_mses = zip(*[(m["client_id"], m["mse"], m["noise_var"], m["clean_mse"]) for _, m in metrics])
    return {
        "client_mse": dict(zip(client_ids, mses)),
        "client_noise_var": dict(zip(client_ids, noise_vars)),
        "client_clean_mse": dict(zip(client_ids, clean_mses))
    }


# System

def wait_for_server(server_address, timeout=600, interval=5):
    server_host, server_port = server_address.split(':')
    server_port = int(server_port)
    start_time = time.time()
    while True:
        try:
            with socket.create_connection((server_host, server_port), timeout=interval):
                break
        except (socket.error, ConnectionRefusedError):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Could not connect to server {server_address} within {timeout} seconds")
            time.sleep(interval)
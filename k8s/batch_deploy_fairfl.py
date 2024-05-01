#!/usr/bin/env python

import os
import argparse
import subprocess
import tempfile
import time

# Default values for optional arguments
DEFAULTS = {
    "n_rounds": 80,
    "n_epochs": 1,
    "batch_size": 16,
    "learning_rate": 0.07,
    "min_available_clients": 8,
    "min_fit_clients": 8,
    "fraction_fit": 1.0,
    "fraction_evaluate": 1.0,
    "abs_fair": 0,
    "tilt": 0.0,
    "norm_lambda": 0.5,
    "recollection_strategy": "uniform",
    "recollection_size_strategy": "constant",
    "recollection_size_default": 128,
}

# FL common arguments
COMMON_ARGUMENTS = [
    "n_rounds",
    "n_epochs",
    "batch_size",
    "learning_rate",
    "min_available_clients",
    "min_fit_clients",
    "fraction_fit",
    "fraction_evaluate",
    "recollection_strategy",
    "recollection_size_strategy",
    "recollection_size_default",
]

# FL algorithm-specific arguments
ALGORITHMS = {
    "local": [],
    "fedha": [],
    "fedhaf": ["tilt", "norm_lambda", "abs_fair"]
}

script_dir = os.path.dirname(os.path.realpath(__file__))

def create_config_map(args, data_file):
    algorithm_args = COMMON_ARGUMENTS + ALGORITHMS[args.algorithm]
    config_data = [f"FAIRFL_CONFIG_{k.upper()}: '{str(getattr(args, k))}'" for k in algorithm_args]
    config_str = "\n  ".join(config_data)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp:
        temp.write(f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: fairfl-config
data:
  FAIRFL_CONFIG_ALGORITHM: '{args.algorithm}'
  FAIRFL_CONFIG_SIM: '{data_file}'  # <-- This replaces FAIRFL_CONFIG_DATAFILE
  {config_str}
""")
        temp.flush()
        return temp.name

def apply_kubectl(filename, action="apply", force=False):
    filename = os.path.join(script_dir, filename)
    command = ["kubectl", action, "-f", filename]
    if force:
        command.append("--force")
    subprocess.run(command)

def resource_exists(resource_type, resource_name):
    return subprocess.run(["kubectl", "get", resource_type, resource_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

def deploy_configmap(args, data_file):
    configmap_filename = create_config_map(args, data_file)
    apply_kubectl(configmap_filename)

def deploy_server():
    deployment_name = "fairfl-server-deployment"
    deployment_exists = resource_exists("deployment", deployment_name)
    if deployment_exists:
        apply_kubectl("deployment/fairfl_server.yaml", action="replace", force=True)
    else:
        apply_kubectl("deployment/fairfl_server.yaml")

def job_completed(job_name):
    """Check if a kubernetes job has completed."""
    result = subprocess.run(
        ["kubectl", "get", "job", job_name, "-o", "jsonpath='{.status.conditions[?(@.type==\"Complete\")].status}'"],
        stdout=subprocess.PIPE
    )
    return result.stdout.decode('utf-8').strip() == "'True'"

def deploy_clients():
    job_name = "fairfl-client-job"
    resource_type = "job.batch"
    if resource_exists(resource_type, job_name):
        subprocess.run(["kubectl", "delete", resource_type, job_name], check=True)
    apply_kubectl("deployment/fairfl_client.yaml")
    
    # Wait for the job to complete
    while not job_completed(job_name):
        time.sleep(5)  # Check every 5 seconds

def get_data_files(leaf_dir):
    """
    Construct the list of data file paths based on the predefined folder structure and naming conventions.
    """
    seeds = list(range(1, 11))
    data_files = [os.path.join(leaf_dir, f"seed_{seed}.json") for seed in seeds]
    return data_files


def main(args):
    if args.algorithm not in ALGORITHMS:
        raise ValueError(f"Invalid FL algorithm specified. Allowed values are {', '.join(ALGORITHMS.keys())}.")

    data_files = get_data_files(args.data_folder)

    for data_file in data_files:
        # Create and apply the configmap for the current data file
        deploy_configmap(args, data_file)
        # Apply or replace the server deployment
        deploy_server()
        # Apply or delete and apply the client job
        deploy_clients()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy FairFL server and clients with specified FL algorithm and configuration.")
    parser.add_argument("algorithm", help=f"FL algorithm to run (e.g., {', '.join(ALGORITHMS.keys())}).")
    parser.add_argument("--data_folder", type=str, required=True, help="Root directory containing data files to be processed.")

    for arg, default in DEFAULTS.items():
        parser.add_argument(f"--{arg}", type=type(default), default=default, help=f"{arg} (default: {default})")

    args = parser.parse_args()
    main(args)

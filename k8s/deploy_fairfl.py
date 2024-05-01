#!/usr/bin/env python

import os
import argparse
import subprocess
import tempfile

# Default values for optional arguments
DEFAULTS = {
    "sim": "iidg0",
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
    "out": ""
}

# FL common arguments
COMMON_ARGUMENTS = [
    "sim",
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
    "out"
]

# FL algorithm-specific arguments
ALGORITHMS = {
    "local": [],
    "fedha": [],
    "fedhaf": ["tilt", "norm_lambda", "abs_fair"]
}

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

def create_config_map(args):
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

def deploy_configmap(args):
    configmap_filename = create_config_map(args)
    apply_kubectl(configmap_filename)

def deploy_server():
    deployment_name = "fairfl-server-deployment"
    deployment_exists = resource_exists("deployment", deployment_name)
    if deployment_exists:
        apply_kubectl("deployment/fairfl_server.yaml", action="replace", force=True)
    else:
        apply_kubectl("deployment/fairfl_server.yaml")

def deploy_clients():
    job_name = "fairfl-client-job"
    resource_type = "job.batch"
    if resource_exists(resource_type, job_name):
        subprocess.run(["kubectl", "delete", resource_type, job_name], check=True)
    apply_kubectl("deployment/fairfl_client.yaml")

def main(args):
    if args.algorithm not in ALGORITHMS:
        raise ValueError("Invalid FL algorithm specified. Allowed values are 'fedavg', 'fedprox', and 'ditto'.")

    # Create and apply the configmap
    deploy_configmap(args)

    # Apply or replace the server deployment
    deploy_server()

    # Apply or delete and apply the client job
    deploy_clients()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy FairFL server and clients with specified FL algorithm and configuration.")
    parser.add_argument("algorithm", help="FL algorithm to run (e.g., 'fedavg', 'fedprox', or 'ditto').")

    for arg, default in DEFAULTS.items():
        parser.add_argument(f"--{arg}", type=type(default), default=default, help=f"{arg} (default: {default})")

    args = parser.parse_args()
    main(args)


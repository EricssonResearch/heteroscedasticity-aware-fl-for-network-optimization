import os

from algorithms.fedha.client import fedha_client
from algorithms.fedhaf.client import fedhaf_client
from common.utils import get_config_env_var, set_config_env_var

if __name__ == "__main__":
    algorithm = get_config_env_var("algorithm")
    set_config_env_var("client_id", os.environ.get("JOB_COMPLETION_INDEX")) # Set client_id

    # Run client
    if algorithm == "fedha":
        fedha_client()
    elif algorithm == "fedhaf":
        fedhaf_client()
    else:
        raise ValueError("Algorithm must be one of 'local', 'fedha' or 'fedhaf'.")
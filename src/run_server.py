from algorithms.fedha.server import fedha_server
from algorithms.fedhaf.server import fedhaf_server
from common.utils import get_config_env_var

if __name__ == "__main__":
    algorithm = get_config_env_var("algorithm")
    
    # Run client
    if algorithm == "fedha":
        fedha_server()
    elif algorithm == "fedhaf":
        fedhaf_server()
    else:
        raise ValueError("Algorithm must be one of 'local', 'fedha' or 'fedhaf'.")
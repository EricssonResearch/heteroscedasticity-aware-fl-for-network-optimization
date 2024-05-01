import flwr as fl

from common.utils import mse_alleatoric_var, config_from_env_list, get_config_env_var, save_experiment
from common.data import read_sim_json
from .strategy import FedHA


def main(
        min_available_clients=30, 
        min_fit_clients=10, 
        fraction_fit=0.3, 
        fraction_evaluate=1.0, 
        n_rounds=200,
        sim=""
    ):

    # Define strategy    
    strategy = FedHA(
        evaluate_metrics_aggregation_fn = mse_alleatoric_var, 
        min_fit_clients = min_fit_clients,
        min_available_clients = min_available_clients,
        fraction_fit = fraction_fit,
        fraction_evaluate = fraction_evaluate,
        on_fit_config_fn = lambda curr_round: {"curr_round": curr_round},
        on_evaluate_config_fn = lambda curr_round: {"curr_round": curr_round}
    )

    # Start Flower server
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy
    )

    sim_data = read_sim_json(sim)
    save_experiment(history, sim_data)


def fedha_server():
    params = [
        ("min_available_clients", int),
        ("min_fit_clients", int),
        ("fraction_fit", float),
        ("fraction_evaluate", float),
        ("n_rounds", int),
        ("sim", str)
    ]
    env_vars = config_from_env_list(params)
    main(**env_vars)


if __name__ == "__main__":
    main()

from collections import OrderedDict
from typing import Dict, Tuple, Callable, Union, Optional

import flwr as fl
import torch
from flwr.common.typing import NDArrays, Scalar

from common.utils import config_from_env_list, wait_for_server, configure_recollection, var_reciprocal
from common.data import read_sim_json, client_sim, ClientSimulationManager
from algorithms.fedha.model import Net, train, val


class Client(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: str,
        net: torch.nn.Module,
        sim_manager: ClientSimulationManager,
        device: torch.device,
        n_epochs: int,
        batch_size: int,
        learning_rate: float,
        recollection_strategy: Tuple[Optional[Callable], bool],
        recollection_size_strategy: Union[Callable, int, None],
    ):
        self.id = client_id
        self.net = net
        self.sim_manager = sim_manager
        self.device = device
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.recollection_strategy = recollection_strategy
        self.recollection_size_strategy = recollection_size_strategy

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Returns the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Changes the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implements distributed fit function for a given client."""
        self.set_parameters(parameters)
        self.sim_manager.set_fl_round(config["curr_round"] - 1)

        trainloader = self.sim_manager.get_trainloader(
                sample_prioritization=self.recollection_strategy,
                recollection_size=self.recollection_size_strategy,
                batch_size=self.batch_size
            )

        train(
            self.net,
            trainloader,
            self.device,
            epochs=self.n_epochs,
            learning_rate=self.learning_rate,
        )
        return (self.get_parameters({}), 
                len(trainloader.dataset), 
                {"sum_sample_weights": torch.sum(var_reciprocal(trainloader.dataset.noise_vars)).item()})

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implements distributed evaluation for a given client."""
        self.set_parameters(parameters)
        self.sim_manager.set_fl_round(config["curr_round"] - 1)

        valloader = self.sim_manager.get_valloader(batch_size=self.batch_size)
        valloss = val(self.net, valloader, self.device)
        
        testloader = self.sim_manager.get_testloader(batch_size=self.batch_size)
        testloss = val(self.net, testloader, self.device)
        
        return (
            float(valloss), 
            len(valloader.dataset), 
            {
                "client_id": self.id, 
                "mse": float(valloss), 
                "noise_var": float(valloader.dataset.mean_noise_var),
                "clean_mse": float(testloss)
            }
        )


def main(
        server_address = "fairfl-server-service:8080",
        sim = "/proj/fair-ai/fair-fl/simulation/sim_iid.json",
        client_id = "0",
        net = Net([19, 10]),
        n_rounds = 200,
        n_epochs = 20,
        batch_size = 32,
        learning_rate = 0.01,
        recollection_strategy = "uniform",
        recollection_size_strategy = "constant",
        recollection_size_default = 100,
    ):

    # Choose device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load simulation
    sim_data = read_sim_json(sim)
    sim_manager = client_sim(sim_data, client_id, n_rounds)

    # Configure recollection
    recollection_strategy, recollection_size_strategy = configure_recollection(recollection_strategy, 
                                                                           recollection_size_strategy, 
                                                                           recollection_size_default)                                                                       

    # Start Flower client
    wait_for_server(server_address)

    fl.client.start_numpy_client(
        server_address=server_address,
        client=Client(
            client_id = client_id,
            net = net,
            sim_manager = sim_manager,
            device = device,
            n_epochs = n_epochs,
            batch_size = batch_size,
            learning_rate = learning_rate,
            recollection_strategy = recollection_strategy,
            recollection_size_strategy = recollection_size_strategy,
        )
    )


# Run in kubernetes with configuration from environment variables
def fedha_client():
    params = [
        ("server_address", str),
        ("sim", str),
        ("client_id", str),
        ("n_rounds", int),
        ("n_epochs", int),
        ("batch_size", int),
        ("learning_rate", float),
        ("recollection_strategy", str),
        ("recollection_size_strategy", str),
        ("recollection_size_default", int),
    ]
    env_vars = config_from_env_list(params)
    main(**env_vars)


if __name__ == "__main__":
    main()
    
import json
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import WeightedRandomSampler

class SimulationData:
    def __init__(self,
        Xtrain: torch.FloatTensor,
        ytrain: torch.FloatTensor,
        Xval: torch.FloatTensor,
        yval: torch.FloatTensor,
        Xtest: torch.FloatTensor,
        ytest: torch.FloatTensor,
        load_schedule: torch.FloatTensor,
        noise_var_schedule: torch.FloatTensor,
        sim_len: int
    ):
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xval = Xval
        self.yval = yval
        self.Xtest = Xtest
        self.ytest = ytest
        self.load_schedule = load_schedule
        self.noise_var_schedule = noise_var_schedule
        self.n_checkpoints = load_schedule.shape[0]
        self.n_train_checkpoint_samples = Xtrain.shape[1]
        self.n_val_checkpoint_samples = Xval.shape[1]
        self.n_dimensions = Xtrain.shape[2]
        self.sim_len = sim_len
        self.retention = 80
        self.cp_sim_len = self.n_checkpoints-self.retention
        self.rounds_per_checkpoint = self.sim_len/self.cp_sim_len


class MemoryBank(Dataset):
    def __init__(self, 
        data: torch.FloatTensor, 
        targets: torch.FloatTensor, 
        noise_var_history: torch.FloatTensor,
    ):
        self.data = data
        self.targets = targets
        self.noise_var_history = noise_var_history
        self.n_checkpoints = data.shape[0]
        self.n_checkpoint_samples = data.shape[1]
        self.n_dimensions = data.shape[2]

    def __len__(self):
        return self.n_checkpoints * self.n_checkpoint_samples

    def __getitem__(self, index):
        ix = np.unravel_index(index, shape=(self.n_checkpoints, self.n_checkpoint_samples))
        return self.data[ix], self.targets[ix], self.noise_var_history[ix[0]]

    def recollect(self, prioritization, retrieval_size):
        weights = prioritization(self)
        n_samples = retrieval_size(self) if callable(retrieval_size) else retrieval_size
        n_samples = min(n_samples, len(weights))
        retrieved_ix = list(WeightedRandomSampler(weights, num_samples=n_samples, replacement=False))
        return Recollection(self, retrieved_ix)
    

class Recollection(Dataset):
    def __init__(self, memorybank: MemoryBank, retrieval_ix: Sequence[int]):
        self.data, self.targets, self.noise_vars = Subset(memorybank, retrieval_ix)[:]
        self.mean_noise_var = self.noise_vars.mean()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class ClientSimulationManager:
    def __init__(self, sim_data: SimulationData):
        self.sim_data = sim_data
        self.fl_round = 0
        
    def set_fl_round(self, fl_round):
        self.fl_round = fl_round
        
    def get_current_checkpoint(self):
        return int(self.fl_round/self.sim_data.rounds_per_checkpoint) + self.sim_data.retention
    
    def get_current_load(self):
        return self.sim_data.load_schedule[self.get_current_checkpoint()].item()
    
    def get_train_memorybank(self):
        ix_newest = self.get_current_checkpoint()
        ix_oldest = ix_newest - self.sim_data.retention
        return MemoryBank(
            data = self.sim_data.Xtrain[ix_oldest:ix_newest+1],
            targets=self.sim_data.ytrain[ix_oldest:ix_newest+1],
            noise_var_history=self.sim_data.noise_var_schedule[ix_oldest:ix_newest+1],
        )
    
    def get_val_memorybank(self):
        ix_newest = self.get_current_checkpoint()
        ix_oldest = ix_newest - self.sim_data.retention
        return MemoryBank(
            data = self.sim_data.Xval[ix_oldest:ix_newest+1],
            targets=self.sim_data.yval[ix_oldest:ix_newest+1],
            noise_var_history=self.sim_data.noise_var_schedule[ix_oldest:ix_newest+1],
        )
        
    def get_trainloader(self, sample_prioritization=None, recollection_size=None, batch_size=32):
        memorybank = self.get_train_memorybank()
        recollection = memorybank.recollect(sample_prioritization, recollection_size)
        return DataLoader(recollection, batch_size=batch_size, shuffle=True)

    def get_valloader(self, batch_size=32):
        return DataLoader(
            Testset(self.sim_data.Xval, 
                    self.sim_data.yval,
                    self.sim_data.noise_var_schedule[-self.sim_data.cp_sim_len:].mean()),
            batch_size=batch_size,
            shuffle=True
        )
    
    def get_testloader(self, batch_size=32):
        return DataLoader(
            Testset(self.sim_data.Xtest, self.sim_data.ytest, 0), 
            batch_size=batch_size, 
            shuffle=True
        )
    

class Testset(Dataset):
    def __init__(self, data, targets, mean_noise_var):
        self.data = data
        self.targets = targets
        self.mean_noise_var = mean_noise_var

    def __len__(self):
        return self.targets.shape[0]
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def client_sim(sim_dict, client_id, sim_len):
    cid = str(client_id)
    
    client_load = torch.tensor(sim_dict["load_schedules"][cid])
    client_noise_var = torch.tensor(sim_dict["noise_var_schedules"][cid])
    
    Xtrain = torch.tensor(sim_dict["data"][cid]["train"]["x"])
    ytrain = torch.tensor(sim_dict["data"][cid]["train"]["y"])
    Xval = torch.tensor(sim_dict["data"][cid]["val"]["x"])
    yval = torch.tensor(sim_dict["data"][cid]["val"]["y"])
    Xtest = torch.tensor(sim_dict["data"][cid]["test"]["x"])
    ytest = torch.tensor(sim_dict["data"][cid]["test"]["y"])

    sim_data = SimulationData(Xtrain, ytrain, Xval, yval, Xtest, ytest, client_load, client_noise_var, sim_len)
    
    return  ClientSimulationManager(sim_data)


def read_sim_json(sim_path):
    preset = get_sim_preset(sim_path)
    sim_path = preset if preset else sim_path

    with open(sim_path, "r") as sim_file:
        sim_dict = json.load(sim_file)

    return sim_dict


def get_sim_preset(preset):
    return f"/proj/fair-ai/fair-fl/simulation/{preset}.json" if preset in PRESETS else None

PRESETS = [
    "bike_hom",
    "bike_het"
]

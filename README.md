# Heteroscedasticity-Aware Federated Learning for Network Optimization

This repository contains the code for the master's thesis "Fair and Efficient Federated Learning for Network Optimization with Heteroscedastic Data".


## Data

The preprocessed data is available in the `simulation` directory. The `jupyter/data-prep` directory contains more information about data sources, transformations and formats.


## Preparation

All experiments were run on Ericsson's kubernetes infratructure, so some preparations will be needed to set up your environment before running the trials.

1. To create the docker image, based on the src folder, and push it to your container registry, run `k8s/docker/update_container.py fairfl-cpu --src`. Make sure to first edit `k8s/docker/update_container.py` to use the address of your container registry.
2. Copy the `simulation` directory to a location that is accessible to your kubernetes job. The directory contains the information needed for each simulated FL client (client pod created by the experiment job) to know its load and available data during the simulation.


## Running experiments

Use the `deploy_fairfl.py` script to run experiments using different simulations and algorithms.

* To run an experiment using FedHA: `k8s/deploy_fairfl.py fedha --sim=<path-to-specific-simulation-json-file> --out=<path-to-output-folder> --learning_rate=<your-learning-rate> --batch_size=<your-batch-size>`
* To run an experiment using FedHAF: `k8s/deploy_fairfl.py fedhaf --sim=<path-to-specific-simulation-json-file> --out=<path-to-output-folder> --learning_rate=<your-learning-rate> --batch_size=<your-batch-size> --recollection_strategy=reciprocal --norm_lambda=0.5 --abs_fair=0 --tilt=<your-tilt>`
* To run an experiment using TERM: `k8s/deploy_fairfl.py fedhaf --sim=<path-to-specific-simulation-json-file> --out=<path-to-output-folder> --learning_rate=<your-learning-rate> --batch_size=<your-batch-size> --recollection_strategy=uniform --norm_lambda=0.5 --abs_fair=1 --tilt=<your-tilt>`
* To run an experiment using FedAvg: `k8s/deploy_fairfl.py fedhaf --sim=<path-to-specific-simulation-json-file> --out=<path-to-output-folder> --learning_rate=<your-learning-rate> --batch_size=<your-batch-size> --recollection_strategy=uniform --tilt=0`


## Viewing results

To get a summary of the experiment results, use the notebook `describe_experiments.ipynb` in the `jupyter` directory.


## Cite this work

```tex
 @masterthesis{Welander_2024, 
    title={Fair and Efficient Federated Learning for Network Optimization with Heteroscedastic Data}, 
    url={https://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-531747}, 
    author={Welander, Andreas}, 
    year={2024},
    month = {June},
    school = {Uppsala University},
    type = {Master's thesis}
}
```
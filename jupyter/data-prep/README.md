# Data Description

# Data Description

The raw Bike Sharing dataset used for the thesis can be downloaded from [UC Irvine's Machine Learning Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset). The repository also contains a detailed description of the dataset features.

The preprocessing of the data is described in section 5.1 (Experimental Setup) of the thesis report, and the exact method can be replicated using the `prep_data` notebook in this directory. Unfortunately, the raw data used to create the load schedules for the simulated Base Stations is proprietary and cannot be shared.

The preprocessed data is available in the `simulation` directory – `bike_het` containing heterogeneous simulations, and `bike_hom` containing homogeneous. Each of these contains simulations for different SNRs (the reported value, NSR, is the inverse of the SNR). For each SNR, there are 11 independent simulations.

The simulations are stored as JSON objects with the following structure:


```
{
  "load_schedules": {
    "<client_index>": [
      "<schedule_value_1>",
      "<schedule_value_2>",
      ...
    ],
    ...
  },
  "noise_var_schedules": {
    "<client_index>": [
      "<noise_variance_value_1>",
      "<noise_variance_value_2>",
      ...
    ],
    ...
  },
  "data": {
    "<client_index>": {
      "train": {
        "x": [
          [
            "<feature_vector_1>",
            "<feature_vector_2>",
            ...
          ],
          [
            "<feature_vector_1>",
            "<feature_vector_2>",
            ...
          ],
          ...
        ],
        "y": [
          [
            "<target_value_1>",
            "<target_value_2>",
            ...
          ],
          [
            "<target_value_1>",
            "<target_value_2>",
            ...
          ],
          ...
        ]
      },
      "val": {
        "x": [
          [
            "<feature_vector_1>",
            "<feature_vector_2>",
            ...
          ],
          [
            "<feature_vector_1>",
            "<feature_vector_2>",
            ...
          ],
          ...
        ],
        "y": [
          [
            "<target_value_1>",
            "<target_value_2>",
            ...
          ],
          [
            "<target_value_1>",
            "<target_value_2>",
            ...
          ],
          ...
        ]
      },
      "test": {
        "x": [
          [
            "<feature_vector_1>",
            "<feature_vector_2>",
            ...
          ],
          [
            "<feature_vector_1>",
            "<feature_vector_2>",
            ...
          ],
          ...
        ],
        "y": [
          [
            "<target_value_1>",
            "<target_value_2>",
            ...
          ],
          [
            "<target_value_1>",
            "<target_value_2>",
            ...
          ],
          ...
        ]
      }
    },
    ...
  }
}
```


Explanation of fields:
* `load_schedules` – the normalized ([0, 1]) load of each individual client at each checkpoint.
* `noise_var_schedules` - the variance of the aleatoric noise of each client at each checkpoint.
* `data` - training, validation, and test data, grouped by checkpoint availability.

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd


@dataclass
class FederatedConfig:
    learning_rate: float
    aggregation_type: str

@dataclass
class FedDPConfig(FederatedConfig):
    num_rounds: int
    epochs: int
    fed_dp_constant: float = 2
    fed_dp_epsilon: float = 0.5


@dataclass
class FedPerConfig(FederatedConfig):
    number_layer_agg: int

@dataclass
class FederatedData:
    clients_data: dict
    x_gen_test: Union[None, pd.DataFrame, np.array]
    y_gen_test: Union[None, pd.DataFrame, np.array]


@dataclass
class FedLearnOptions:
    multithreaded: bool = False
    round_to_start: int = 0
    snapshot_frequency: int = 5


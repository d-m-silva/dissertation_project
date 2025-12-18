from copy import deepcopy

import numpy as np

from .fed_agg import FedAgg
from .. import FederatedConfig
from ..config import FedPerConfig


class FedPer(FedAgg):

    def __init__(self, clients_name: list[str], init_weights: np.ndarray, client_train_sizes: dict[str, int]):
        super().__init__(clients_name, init_weights, client_train_sizes)

        self.number_layer_agg = 0

    def add_config(self, config: FedPerConfig) -> None:
        self.number_layer_agg = config.number_layer_agg

    def aggregate_updates(self, updates: dict[str, np.ndarray]) -> np.ndarray:
        out_weights = deepcopy(self.agg_weights)

        for n_layer in range(self.number_layer_agg):
            layer = self.agg_weights[n_layer]

            layer += np.median(np.stack([updates[client][n_layer] for client in self.clients_name]), axis=0)
            out_weights[n_layer] = layer

        return out_weights

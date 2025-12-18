from copy import deepcopy

import numpy as np
import tensorflow as tf

from . import FedAvg
from ..config import FedDPConfig


class FedDP(FedAvg):

    def __init__(self, clients_name: list[str], init_weights: np.ndarray, client_train_sizes: dict[str, int]):

        super().__init__(clients_name, init_weights, client_train_sizes)
        self.std_layer_coefficient = 0

    def add_config(self, config: FedDPConfig) -> None:
        min_train_size = min(list(self.client_train_sizes.values()))
        nb_client = len(self.clients_name)

        std_under_sqrt = abs(config.num_rounds ** 2 - (config.epochs ** 2) * nb_client)
        std_denominator = min_train_size * nb_client * config.fed_dp_epsilon

        self.std_layer_coefficient = 2 * config.fed_dp_constant * np.sqrt(std_under_sqrt) / std_denominator

    def aggregate_updates(self, updates: dict[str, np.ndarray]) -> np.ndarray:
        avg_aggr_weights = super().aggregate_updates(updates)

        dp_noise = deepcopy(avg_aggr_weights)

        for n_layer in range(len(self.agg_weights)):

            normed_updates_layer = []
            for client in self.clients_name:
                normed_updates_layer.append(np.linalg.norm(updates[client][n_layer]))

            max_update = np.max(normed_updates_layer)  # Called C in the paper
            std_layer = np.zeros(avg_aggr_weights[n_layer].shape) + self.std_layer_coefficient * max_update
            mean_layer = np.zeros(avg_aggr_weights[n_layer].shape)
            noise_layer = tf.random.normal(avg_aggr_weights[n_layer].shape, mean=mean_layer, stddev=std_layer)
            dp_noise[n_layer] += noise_layer

        return dp_noise

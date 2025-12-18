from copy import deepcopy

import numpy as np

from .fed_agg import FedAgg


class FedAvg(FedAgg):

    def aggregate_updates(self, updates: dict[str, np.ndarray]) -> np.ndarray:
        out_weights = deepcopy(self.agg_weights)

        for n_layer in range(len(self.agg_weights)):
            layer = self.agg_weights[n_layer]

            layer += np.average(np.stack([updates[client][n_layer] for client in self.clients_name]),
                                weights=[self.client_train_sizes[client] for client in self.client_train_sizes.keys()],
                                axis=0)

            out_weights[n_layer] = layer

        return out_weights

from abc import ABC, abstractmethod

import numpy as np

from ..config import FederatedConfig


class FedAgg(ABC):

    def __init__(self, clients_name: list[str], init_weights: np.ndarray, client_train_sizes: dict[str, int]) -> None:
        self.plus_agg = True
        self.clients_name = clients_name

        self.agg_weights = init_weights
        self.client_train_sizes = client_train_sizes

    def add_config(self, config: FederatedConfig) -> None:
        pass

    def add_clients(self, clients_name: list[str], clients_train_sizes: dict[str, int]) -> None:
        self.clients_name += clients_name
        self.client_train_sizes = {**self.client_train_sizes, **clients_train_sizes}

    @abstractmethod
    def aggregate_updates(self, updates: dict[str, np.ndarray]):
        raise NotImplemented("Aggregation method should be implemented in sub classes")

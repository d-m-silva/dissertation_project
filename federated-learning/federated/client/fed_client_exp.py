from copy import deepcopy
from typing import Callable

import numpy as np



class FedClient:

    def __init__(self, name: str, data: dict[str, np.ndarray], model: Callable, learning_rate: float) -> None:
        self.name = name

        self.model = model(learning_rate)

        # Subset for FL training
        self.x_train = data['fl']['x_train']
        self.y_train = data['fl']['y_train']
        self.x_val = data['fl']['x_val']
        self.y_val = data['fl']['y_val']

        #Subset for local Finetune
        self.x_all = data['all']['x']
        self.y_all = data['all']['y']

        self.history_list = []
        self.weights_before_update = self.model.get_weights()

        self._update = []

    def update_client(self, server_update: list) -> None:
        weights = self.model.get_weights()
        self.weights_before_update = deepcopy(weights)

        for num_layer in range(len(weights)):
            weights[num_layer] += server_update[num_layer]

        self.model.set_weights(weights)

    def train_client(self, epochs: int, batch_size: int) -> None:
        history = self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(self.x_val, self.y_val), verbose=0)

        self.history_list.append(history)

        weights_after_update = self.model.get_weights()
        client_update = []

        for num_layer in range(len(weights_after_update)):
            client_update.append(
                self.weights_before_update[num_layer] - weights_after_update[num_layer])

        self._update = client_update

    def get_update_client(self) -> list:
        return self._update

    def evaluate_client(self) -> float:
        _, acc_client = self.model.evaluate(self.x_val, self.y_val, verbose=0)

        return acc_client

    def evaluate_generalization_model(self, x_gen, y_gen) -> float:
        _, acc_client = self.model.evaluate(x_gen, y_gen, verbose=0)

        return acc_client


    #Local finetune training
    def train_with_all_data(self, epochs: int, batch_size: int) -> None:
        x_all = self.x_all
        y_all = self.y_all

        history = self.model.fit(
            x_all, y_all, epochs=epochs, batch_size=batch_size, verbose=0)
        self.history_list.append(history)

    #Local finetune evaluation
    def evaluate_client_with_all_data(self) -> float:

        _, acc = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        return acc

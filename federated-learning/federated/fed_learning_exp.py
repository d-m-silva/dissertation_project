import importlib
import logging
import pickle
import threading
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

from .client.fedclient_exp import FedClient
from .config import FederatedConfig, FederatedData


class FedLearning:

    def __init__(self, data: FederatedData, model: Callable, clients_name: list[str], config: FederatedConfig, output_suffix: str = "") -> None:
        self.aggregated_model = model(config.learning_rate)

        self.x_gen_test = data.x_gen_test
        self.y_gen_test = data.y_gen_test

        self.agg_weights = self.aggregated_model.get_weights()

        clients_data = data.clients_data
        self.clients = [FedClient(client_name, clients_data.get(client_name), model, config.learning_rate) for
                        client_name in clients_name]

        client_train_sizes = {client_name: len(clients_data.get(client_name, {}).get("fl", {}).get("x_val", []))
        for client_name in clients_name}

        aggregation_type = config.aggregation_type
        try:
            module = importlib.import_module("federated.server")
            self.server = getattr(module, aggregation_type)(clients_name, self.aggregated_model.get_weights(),
                                                            client_train_sizes)
        except AttributeError as e:
            raise ModuleNotFoundError(e)

        self.server.add_config(config)

        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger(__name__)

        self._output_filename_prefix = f"{aggregation_type}_C{len(self.clients)}"

        #Adding a customisable suffix, to help track experiment results

        if output_suffix:
           self._output_filename_prefix += f"_{output_suffix}"

    def evaluate_model(self) -> tuple[dict, int]:
        client_accuracies = dict()
        generalization_accuracy = 0

        for client in self.clients:
            acc_client = client.evaluate_client()
            client_accuracies[client.name] = acc_client

            self._logger.debug(
                f"Aggregated model precision, Client {client.name}: {acc_client}")

        self._logger.info(
            f"Average model precision: {sum(client_accuracies.values()) / len(client_accuracies)}")

        if self.x_gen_test is not None and self.y_gen_test is not None:
            for client in self.clients:
                generalization_accuracy += client.evaluate_generalization_model(
                    self.x_gen_test, self.y_gen_test)

            self._logger.info(
                f"Generalization model precision: {generalization_accuracy / len(self.clients)}")

        return client_accuracies, generalization_accuracy

    def snapshot_models(self, round_number: int) -> None:
        self._logger.info(f"Snapshot models for round {round_number}")
        path = Path(f"{self._output_filename_prefix}/round_{round_number}")
        path.mkdir(parents=True, exist_ok=True)
        for client in self.clients:
            client.model.save_weights(f"{path}/{client.name}_weights.h5")

    def load_models(self, round_number: int) -> None:
        self._logger.info(f"Load models from round {round_number}")
        for client in self.clients:
            client.model.load_weights(
                f"{self._output_filename_prefix}/round_{round_number}/{client.name}_weights.h5")

    def _compute_round(self, epochs: int, batch_size: int, multithreaded: bool):
        # Federated Learning : client side
        if multithreaded:
            threads = []
            for client in self.clients:
                client.update_client(self.agg_weights)
                thread = threading.Thread(
                    target=client.train_client, args=[epochs, batch_size])
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

        updates = dict()
        for client in self.clients:
            if not multithreaded:
                client.update_client(self.agg_weights)
                client.train_client(epochs, batch_size)
            updates[client.name] = deepcopy(client.get_update_client())

        # Federated Learning : Server side
        self.agg_weights = self.server.aggregate_updates(updates)

    def run(self, epochs: int, batch_size: int, num_rounds: int, finetune_after_fl: bool = True, resume_round: int = 0) -> tuple[dict, ndarray]:

        round_to_start = resume_round

        num_new_rounds = num_rounds - round_to_start

        accuracies = dict((client.name, np.zeros(num_new_rounds))
                          for client in self.clients)
        generalization_acc = np.zeros(num_new_rounds)

        self._output_filename_prefix += f"_N{num_rounds}_E{epochs}_B{batch_size}"
        snapshot_frequency = 5

        if round_to_start > 0:
            self.load_models(round_to_start)

        for round_number in range(round_to_start, num_rounds):
            print()
            print("#" * 70)
            print("#" * 25 + f"{f'Learning round {round_number + 1}': ^20}" + "#" * 25)
            print("#" * 70)
            print()

            # Federated Learning : client side and server side
            self._compute_round(epochs, batch_size, run_options)

            # Federated Learning : Aggregated model evaluation
            client_acc, gen_acc = self.evaluate_model()
            idx = round_number - round_to_start
            for client in self.clients:
                accuracies[client.name][idx] = client_acc[client.name]
            generalization_acc[idx] = gen_acc

            if (round_number + 1) % snapshot_frequency == 0 and round_number != round_to_start:
                self.snapshot_models(round_number + 1)

        # Local finetune after FL
        finetune_accuracies = dict()
        finetune_gen_acc = 0
        finetune_epochs = 5

        if finetune_after_fl:

            print()
            print("#" * 70)
            print("#" * 20 + "Fine-tuning with full local data " + "#" * 20)
            print("#" * 70)

            for client in self.clients:
                client.update_client(self.agg_weights)
                client.train_with_all_data(
                    epochs=finetune_epochs, batch_size=batch_size)
                acc = client.evaluate_client_with_all_data()
                finetune_accuracies[client.name] = acc

            if self.x_gen_test is not None and self.y_gen_test is not None:
                for client in self.clients:
                    finetune_gen_acc += client.evaluate_generalization_model(
                        self.x_gen_test, self.y_gen_test
                    )

                finetune_gen_acc /= len(self.clients)

            self._logger.info(
                f"Final client accuracies after fine-tuning: {finetune_accuracies}")
            self._logger.info(
                f"Generalization accuracy after fine-tuning: {finetune_gen_acc}")

        return accuracies, generalization_acc, finetune_accuracies, finetune_gen_acc

    def plot_cumulative_accuracy(self, accuracies: dict, directory: str, show: bool = False) -> None:
        final_round_accuracy = []

        for client in self.clients:
            # Get accuracy of the final round for each state
            final_round_accuracy.append(accuracies[client.name][-1])

        sorted_accuracy = np.sort(final_round_accuracy)
        nb_clients = len(sorted_accuracy)

        cumulative_counts = np.array(
            range(1, nb_clients + 1)) * 100 / nb_clients

        plt.plot(sorted_accuracy, cumulative_counts)
        plt.xlabel("Accuracy on local data", fontsize=15)
        plt.ylabel("Cumulative percentage of centers", fontsize=15)
        plt.legend()
        plt.grid(True)

        # Create the directory if it doesn't exist
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        plt.savefig(path / f"{self._output_filename_prefix}_cum_acc_final.png")

        if show:
            plt.show()

        plt.clf()

    def plot_model(self, accuracies: dict, generalization_accuracies: np.ndarray, num_rounds: int, directory: str,
                   show: bool = False) -> None:
        list_rounds = list(range(num_rounds))
        list_mean = [np.mean(np.array(list(accuracies.values()))[:, i])
                     for i in list_rounds]

        plt.plot(list_rounds, list_mean, label='Mean accuracy on local data')
        plt.plot(list_rounds, generalization_accuracies,
                 label='Generalization accuracy')
        plt.xlabel("Learning rounds", fontsize=15)
        plt.ylabel("Accuracy", fontsize=15)
        plt.legend()
        plt.grid(True)

        # Create the directory if it doesn't exist
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        images_path = path / "pictures"
        images_path.mkdir(parents=True, exist_ok=True)

        plt.savefig(
            images_path / f"{self._output_filename_prefix}_mean_acc_evol.png")

        if show:
            plt.show()

        self.plot_cumulative_accuracy(accuracies, directory, show)

    def store_accuracies(self, accuracies: dict, generalization_accuracies: np.ndarray, directory: str, filename_prefix: str = None) -> None:
        # Create the directory if it doesn't exist
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        accuracies_path = path / "accuracies"
        accuracies_path.mkdir(parents=True, exist_ok=True)

        filename = filename_prefix if filename_prefix else self._output_filename_prefix
        with open(accuracies_path / f"{filename}_acc.sav", "wb") as file:
            pickle.dump(accuracies, file)

        with open(accuracies_path / f"{filename}_gen_acc.sav", "wb") as file:
            pickle.dump(generalization_accuracies, file)

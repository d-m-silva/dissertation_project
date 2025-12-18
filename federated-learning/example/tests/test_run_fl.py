import argparse
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.expanduser("~/federated-learning"))

from config_exp import get_config
import tensorflow as tf

from federated.fed_learning_exp import FedLearning
from federated.fed_attack import FedAttack
from federated.feddata.adult_fl_loader import load_acs_local_data
from federated.feddata.acs_data_states import ACSDataStatesBySize
from model import model_adult
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Script")
    parser.add_argument("--aggregation-type", type=str, default="FedAvg",
                        choices=["FedAvg", "FedMed", "FedDP", "FedPer", "FedThr", "FedTMean"],
                        help="Choose the federated learning aggregation algorithm")
    parser.add_argument("--num-rounds", type=int, default=5, help="Number of learning rounds")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs per client per learning round")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--clients-number", type=int, default=10, help="Number of clients")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--size-agg", type=int, default=10, help="Size of aggregation for FedPer")
    parser.add_argument("--multithread", action='store_true', help="Use multithreading for client update")
    parser.add_argument('--client-number-agg', action='store_true',
                        help="Specify if the aggregation is made with the number of each client instead of data size")
    parser.add_argument("--finetune_after_fl", action = 'store_true', help = "Fine-tune after FL")
    parser.add_argument('--resume-round', type=int, default=0, help='Round from where fine-tuning restarts')
    parser.add_argument('--outlier_cap', type = float, default = 1, help='Percentage of outliers used, from around 0 to 1 (all)')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    # Setup initialization
    config = get_config(args)

    # Get client's names
    clients_name = [state.name for state in ACSDataStatesBySize][:args.clients_number]

    # Generate local datasets
    data = load_acs_local_data(list_states=clients_name, use_subset_criteria = "cosine_similarity_thr_2_5_pct")

    output_suffix = "Cosine_Similarity_Frequency_Encoding_thr_2_5_pct" #"IQR"

    if args.outlier_cap != 1:
       output_suffix += f"_{args.outlier_cap}_outlier_cap"

    # Model initialization
    model = model_adult

    federated_instance = FedLearning(data, model, clients_name, config, output_suffix)

    accuracies, gen_accuracies, finetune_accuracies, finetune_gen_acc = federated_instance.run(args.epochs, args.batch_size, args.num_rounds, args.multithread, args.finetune_after_fl, resume_round=args.resume_round)

    # Store accuracies
    federated_instance.store_accuracies(accuracies, gen_accuracies, args.output_dir)

    if args.finetune_after_fl:
        finetune_prefix = federated_instance._output_filename_prefix + "_Finetune"
        federated_instance.store_accuracies(finetune_accuracies, finetune_gen_acc, args.output_dir, filename_prefix=finetune_prefix)


if __name__ == "__main__":
    main()

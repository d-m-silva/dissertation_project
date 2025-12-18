from federated import FederatedConfig


def get_config(args):
    config = FederatedConfig(learning_rate=args.learning_rate, aggregation_type=args.aggregation_type)


    return config

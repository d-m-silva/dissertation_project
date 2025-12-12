from federated import FederatedConfig
from federated.config import SagFLConfig


def get_config(args):
    config = FederatedConfig(learning_rate=args.learning_rate, aggregation_type=args.aggregation_type)


    return config


def get_attack_config(args):
    config = SagFLConfig(learning_rate=args.learning_rate, learning_rate_ascent=args.learning_rate_ascent,
                         size_agg=args.size_agg)

    return config

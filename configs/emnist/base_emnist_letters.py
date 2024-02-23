import ml_collections


def get_emnist_letters_config(config):
    # training
    config.training.num_steps = 500000
    config.training.eval_freq = 50000
    config.training.print_freq = 1000
    config.training.gamma = "constant"
    config.training.flow_sigma = 0.1

    # data
    config.data = data = ml_collections.ConfigDict()
    data.source = "emnist"
    data.target = "emnist"
    data.shape = [1, 28, 28]
    data.precomputed_stats_file = "emnist_letters"
    data.shuffle_buffer = 10_000
    data.eval_labels = [0, 1, 2]

    return config

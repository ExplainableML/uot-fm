import ml_collections


def get_cifar_config(config):
    config.task = "generation"

    # training
    config.training.flow_sigma = 0.0
    config.training.num_steps = 300000
    config.training.eval_freq = 50000
    config.training.print_freq = 1000

    # data
    config.data = data = ml_collections.ConfigDict()
    data.source = "gaussian"
    data.target = "cifar10"
    data.centered = True
    data.shape = [3, 32, 32]
    data.map_forward = True
    data.precomputed_stats_file = "cifar10_train"
    data.shuffle_buffer = 10_000
    data.eval_paired = False
    data.do_flip = True

    config.eval.labelwise = False
    config.eval.eval_samples = 50000

    return config

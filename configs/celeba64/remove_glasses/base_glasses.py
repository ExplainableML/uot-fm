def get_glasses_config(config):
    # training
    config.training.num_steps = 100000
    config.training.eval_freq = 25000
    config.training.print_freq = 1000
    # data
    config.data.attribute_id = 15
    config.data.map_forward = False
    config.data.precomputed_stats_file = "celeba64_remove-glasses"
    config.data.eval_labels = [17, 20, 201, 35]

    config.eval.labelwise = True

    return config

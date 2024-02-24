from configs.base_config import get_base_config


def get_fm_config():
    config = get_base_config()
    config.t1 = 1.0
    config.t0 = 0.0
    config.dt0 = 0.0
    config.solver = "tsit5"

    # training
    config.training.method = "flow"
    config.training.gamma = "constant"
    config.training.flow_sigma = 0.0
    config.training.matching = False
    config.training.tau_a = 0.0
    config.training.tau_b = 0.0
    config.training.epsilon = 0.0

    return config

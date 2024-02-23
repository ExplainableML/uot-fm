from configs.base_uotfm import get_uotfm_config
from configs.emnist.base_mlpmixer import get_mlpmixer_config
from configs.emnist.base_emnist_letters import get_emnist_letters_config


def get_config():
    config = get_uotfm_config()
    config = get_mlpmixer_config(config)
    config = get_emnist_letters_config(config)
    config.name = "uot-fm_emnist_letters"
    config.training.tau_a = 0.9
    config.training.tau_b = 1.0

    return config

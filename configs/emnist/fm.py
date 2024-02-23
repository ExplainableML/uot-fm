from configs.base_fm import get_fm_config
from configs.emnist.base_mlpmixer import get_mlpmixer_config
from configs.emnist.base_emnist_letters import get_emnist_letters_config


def get_config():
    config = get_fm_config()
    config = get_mlpmixer_config(config)
    config = get_emnist_letters_config(config)
    config.name = "fm_emnist_letters"

    return config

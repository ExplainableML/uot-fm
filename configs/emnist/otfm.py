from configs.base_otfm import get_otfm_config
from configs.emnist.base_mlpmixer import get_mlpmixer_config
from configs.emnist.base_emnist_letters import get_emnist_letters_config


def get_config():
    config = get_otfm_config()
    config = get_mlpmixer_config(config)
    config = get_emnist_letters_config(config)
    config.name = "ot-fm_emnist_letters"

    return config

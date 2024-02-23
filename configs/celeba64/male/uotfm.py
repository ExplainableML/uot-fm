from configs.base_uotfm import get_uotfm_config
from configs.celeba64.base_unet import get_unet_config
from configs.celeba64.base_celeba import get_celeba_config
from configs.celeba64.male.base_male import get_male_config


def get_config():
    config = get_uotfm_config()
    config = get_unet_config(config)
    config = get_celeba_config(config)
    config = get_male_config(config)
    config.name = "uot-fm_celeba64_male"
    config.training.tau_a = 0.95
    config.training.tau_b = 0.95

    return config

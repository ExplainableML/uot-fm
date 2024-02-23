from configs.base_uotfm import get_uotfm_config
from configs.celeba256.base_unet import get_unet_config
from configs.celeba256.base_celeba import get_celeba_config
from configs.celeba256.remove_glasses.base_glasses import get_glasses_config


def get_config():
    config = get_uotfm_config()
    config = get_unet_config(config)
    config = get_celeba_config(config)
    config = get_glasses_config(config)
    config.name = "uot-fm_celeba256_remove_glasses"
    config.training.tau_a = 0.95
    config.training.tau_b = 0.95

    return config

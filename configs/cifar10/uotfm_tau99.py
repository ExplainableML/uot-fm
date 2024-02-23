from configs.base_uotfm import get_uotfm_config
from configs.cifar10.base_unet import get_unet_config
from configs.cifar10.base_cifar import get_cifar_config


def get_config():
    config = get_uotfm_config()
    config = get_unet_config(config)
    config = get_cifar_config(config)
    config.name = "uot-fm_cifar10_tau0.99"
    config.training.tau_a = 0.99
    config.training.tau_b = 0.99

    return config

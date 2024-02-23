from configs.base_otfm import get_otfm_config
from configs.cifar10.base_unet import get_unet_config
from configs.cifar10.base_cifar import get_cifar_config


def get_config():
    config = get_otfm_config()
    config = get_unet_config(config)
    config = get_cifar_config(config)
    config.name = "ot-fm_cifar10"

    return config

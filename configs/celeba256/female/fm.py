from configs.base_fm import get_fm_config
from configs.celeba256.base_unet import get_unet_config
from configs.celeba256.base_celeba import get_celeba_config
from configs.celeba256.female.base_female import get_female_config


def get_config():
    config = get_fm_config()
    config = get_unet_config(config)
    config = get_celeba_config(config)
    config = get_female_config(config)
    config.name = "fm_celeba256_female"

    return config

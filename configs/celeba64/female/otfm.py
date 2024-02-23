from configs.base_otfm import get_otfm_config
from configs.celeba256.base_unet import get_unet_config
from configs.celeba256.base_celeba import get_celeba_config
from configs.celeba256.female.base_female import get_female_config


def get_config():
    config = get_otfm_config()
    config = get_unet_config(config)
    config = get_celeba_config(config)
    config = get_female_config(config)
    config.name = "ot-fm_celeba64_female"

    return config

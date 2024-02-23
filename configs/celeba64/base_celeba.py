import ml_collections


def get_celeba_config(config):
    config.task = "translation"
    config.training.flow_sigma = 0.01
    # data
    config.data = data = ml_collections.ConfigDict()
    data.source = "celeba_attribute"
    data.target = "celeba_attribute"
    data.shape = [3, 64, 64]
    data.centered = True
    data.shuffle_buffer = 10_000
    data.random_crop = False
    data.eval_paired = True

    return config

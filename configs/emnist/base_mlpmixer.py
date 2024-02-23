import ml_collections


def get_mlpmixer_config(config):
    config.training.batch_size = 256
    # model
    config.model = model = ml_collections.ConfigDict()
    model.type = "mlpmixer"
    model.patch_size = 4
    model.hidden_size = 64
    model.mix_patch_size = 512
    model.mix_hidden_size = 512
    model.num_blocks = 4
    model.input_shape = [1, 28, 28]
    model.use_vae = False

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = "adam"
    optim.learning_rate = 3e-4
    optim.beta_one = 0.9
    optim.beta_two = 0.999
    optim.eps = 1e-8
    optim.grad_clip = 1.0
    optim.warmup = 0.0
    optim.schedule = "constant"
    optim.ema_decay = 0.9999

    return config

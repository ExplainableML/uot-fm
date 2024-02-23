import ml_collections


def get_unet_config(config):
    config.training.batch_size = 128
    # model
    config.model = model = ml_collections.ConfigDict()
    model.type = "unet"
    model.hidden_size = 256
    model.dim_mults = [1, 2, 2, 2]
    model.num_res_blocks = 2
    model.heads = 4
    model.dim_head = 64
    model.attention_resolution = [16]
    model.dropout = 0.1
    model.biggan_sample = False
    model.use_vae = False
    model.input_shape = [3, 32, 32]
    model.input_conditional = False

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = "adam"
    optim.learning_rate = 2e-4
    optim.beta_one = 0.9
    optim.beta_two = 0.999
    optim.eps = 1e-8
    optim.grad_clip = 1.0
    optim.warmup = 5000
    optim.schedule = "constant"
    optim.ema_decay = 0.9999

    return config

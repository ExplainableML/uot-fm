import ml_collections


def get_base_config():
    config = ml_collections.ConfigDict()
    config.seed = 42
    config.overfit_to_one_batch = False
    config.wandb_key = None
    config.wandb_group = None
    config.wandb_entity = None

    # training
    config.training = training = ml_collections.ConfigDict()
    training.print_freq = 1000
    training.save_checkpoints = True
    training.preemption_ckpt = False
    training.ckpt_freq = 10000
    training.resume_ckpt = False

    config.eval = eval = ml_collections.ConfigDict()
    eval.compute_metrics = True
    eval.enable_fid = True
    eval.enable_path_lengths = True
    eval.enable_mse = False
    eval.checkpoint_metric = "fid"
    eval.save_samples = True
    eval.num_save_samples = 7
    eval.labelwise = True
    eval.checkpoint_step = None

    return config

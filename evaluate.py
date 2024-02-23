import equinox as eqx
import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.random as jr
import jax.sharding as sharding
import logging
import ml_collections
import numpy as np
import orbax.checkpoint as obx
import wandb

from models import get_model, get_vae_fns
from utils import MetricComputer, get_translation_datasets, get_loss_builder


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    """Evaluation script."""
    jax.config.update("jax_threefry_partitionable", True)
    # create rng keys
    key = jr.PRNGKey(config.seed)
    np.random.seed(config.seed)
    model_key, eval_key = jr.split(key, 2)
    # set up sharding
    num_devices = len(jax.devices())
    # shard needs to have same number of dimensions as the input
    devices = mesh_utils.create_device_mesh((num_devices, 1, 1, 1))
    shard = sharding.PositionalSharding(devices)
    if config.model.use_vae:
        logging.info("Loading VAE...")
        # load vae and jitted encode/decode functions
        vae_encode_fn, vae_decode_fn = get_vae_fns(shard)

    if config.task == "translation":
        _, _, eval_src_ds, eval_tgt_ds = get_translation_datasets(
            config, shard, vae_encode_fn if config.model.use_vae else None
        )
        logging.info(f"num_eval_src: {eval_src_ds.length}")
        logging.info(f"num_eval_tgt: {eval_tgt_ds.length}")
    elif config.task == "generation":
        eval_src_ds, eval_tgt_ds = None, None
    # build model and optimization functions
    model = get_model(config, config.model.input_shape, model_key)
    loss_builder = get_loss_builder(config)
    sample_fn = loss_builder.get_sample_fn()
    # create checkpoint manager
    mngr_options = obx.CheckpointManagerOptions(
        create=True, max_to_keep=3, best_fn=lambda metric: metric, best_mode="min"
    )
    ckpt_mngr = obx.CheckpointManager(
        directory=f"{workdir}/{config.name}/checkpoints",
        checkpointers=obx.Checkpointer(obx.PyTreeCheckpointHandler()),
        options=mngr_options,
    )
    # load saved checkpoint
    if config.eval.checkpoint_step is not None:
        latest_step = config.eval.checkpoint_step
    else:
        latest_step = ckpt_mngr.best_step()
    logging.info(f"Loading model from step {latest_step}...")
    params, static = eqx.partition(model, eqx.is_array)
    restored_ckpt = ckpt_mngr.restore(latest_step, params)
    restored_params = eqx.filter(restored_ckpt, eqx.is_array)
    model = eqx.combine(restored_params, static)
    inference_model = eqx.tree_inference(model, value=True)
    metric_computer = MetricComputer(
        config=config,
        shard=shard,
        eval_ds=eval_src_ds,
        sample_fn=sample_fn,
        vae_encode_fn=vae_encode_fn if config.model.use_vae else None,
        vae_decode_fn=vae_decode_fn if config.model.use_vae else None,
    )
    wandb.login(key=config.wandb_key)
    wandb.init(
        project="uot-fm",
        group=config.wandb_group,
        entity=config.wandb_entity,
        name=f"eval_{config.name}",
        config=config,
    )
    logging.info(f"Computing metrics...")
    eval_dict = metric_computer.compute_metrics(inference_model, eval_key)
    logging.info(f"Metrics: {eval_dict}")
    wandb.log(eval_dict)

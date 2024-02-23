import equinox as eqx
import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.numpy as jnp
import jax.random as jr
import jax.sharding as sharding
import logging
import ml_collections
import numpy as np
import optax
import orbax.checkpoint as obx
import os
from tqdm import tqdm
import wandb

from models import get_model, get_vae_fns
from utils import (
    BatchResampler,
    MetricComputer,
    get_translation_datasets,
    get_generation_datasets,
    get_loss_builder,
    get_optimizer,
)


def train(config: ml_collections.ConfigDict, workdir: str):
    """Training script."""
    jax.config.update("jax_threefry_partitionable", True)
    # create rng keys
    key = jr.PRNGKey(config.seed)
    np.random.seed(config.seed)
    model_key, train_key, eval_key = jr.split(key, 3)
    # set up sharding
    num_devices = len(jax.devices())
    # shard needs to have same number of dimensions as the input
    devices = mesh_utils.create_device_mesh((num_devices, 1, 1, 1))
    shard = sharding.PositionalSharding(devices)
    # get data
    batch_size = config.training.batch_size
    if config.model.use_vae:
        logging.info("Loading VAE...")
        # load vae and jitted encode/decode functions
        vae_encode_fn, vae_decode_fn = get_vae_fns(shard)

    if config.task == "translation":
        train_src_ds, train_tgt_ds, eval_src_ds, eval_tgt_ds = get_translation_datasets(
            config, shard, vae_encode_fn if config.model.use_vae else None
        )
        train_src_loader, train_tgt_loader = iter(train_src_ds), iter(train_tgt_ds)
        logging.info(f"num_train_src: {train_src_ds.length}")
        logging.info(f"num_train_tgt: {train_tgt_ds.length}")
        logging.info(f"num_eval_src: {eval_src_ds.length}")
        logging.info(f"num_eval_tgt: {eval_tgt_ds.length}")
    elif config.task == "generation":
        train_loader = get_generation_datasets(config)
        train_src_ds, train_tgt_ds, eval_src_ds, eval_tgt_ds = None, None, None, None
    if config.training.matching:
        batch_resampler = BatchResampler(
            batch_size=batch_size,
            tau_a=config.training.tau_a,
            tau_b=config.training.tau_b,
            epsilon=config.training.epsilon,
        )
    # build model and optimization functions
    model = get_model(config, config.model.input_shape, model_key)
    loss_builder = get_loss_builder(config)
    loss_fn = loss_builder.get_batch_loss_fn()
    opt = get_optimizer(config)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    train_step_fn = loss_builder.get_train_step_fn(loss_fn, opt.update)
    if config.optim.ema_decay > 0.0:
        assert config.optim.ema_decay < 1.0
        opt_ema = optax.ema(config.optim.ema_decay, debias=False)
        ema_state = opt_ema.init(eqx.filter(model, eqx.is_array))

        @eqx.filter_jit(donate="all-except-first")
        def update_ema(curr_model, curr_ema_state):
            _, ema_state = opt_ema.update(eqx.filter(curr_model, eqx.is_array), curr_ema_state)
            return ema_state

    else:
        ema_state = None

    if config.eval.compute_metrics:
        sample_fn = loss_builder.get_sample_fn()
        metric_computer = MetricComputer(
            config=config,
            shard=shard,
            eval_ds=eval_src_ds,
            sample_fn=sample_fn,
            vae_encode_fn=vae_encode_fn if config.model.use_vae else None,
            vae_decode_fn=vae_decode_fn if config.model.use_vae else None,
        )
        if config.training.save_checkpoints:
            # create checkpoint manager
            mngr_options = obx.CheckpointManagerOptions(
                create=True,
                max_to_keep=3,
                best_fn=lambda metric: metric,
                best_mode="min",
            )
            ckpt_mngr = obx.CheckpointManager(
                directory=f"{os.getcwd()}/{workdir}/{config.name}/checkpoints",
                checkpointers=obx.Checkpointer(obx.PyTreeCheckpointHandler()),
                options=mngr_options,
            )
    if config.training.preemption_ckpt:
        # create checkpoint manager for preemption checkpoints
        preemption_ckpt_mngr = obx.CheckpointManager(
            directory=f"{os.getcwd()}/{workdir}/{config.name}/preemption_ckpt",
            checkpointers=obx.Checkpointer(obx.PyTreeCheckpointHandler()),
            options=obx.CheckpointManagerOptions(create=True, max_to_keep=1),
        )
    steps = config.training.num_steps
    _, static = eqx.partition(model, eqx.is_array)
    if config.training.resume_ckpt:
        # load last saved checkpoint
        resume_step = preemption_ckpt_mngr.latest_step()
        logging.info(f"Resuming training from step {resume_step}...")
        restore_target = {"model": model, "opt_state": opt_state, "opt": opt, "ema_state": ema_state}
        restored_ckpt = preemption_ckpt_mngr.restore(resume_step, restore_target)
        restored_model = eqx.filter(restored_ckpt["model"], eqx.is_array)
        model = eqx.combine(restored_model, static)
        opt_state = restored_ckpt["opt_state"]
        opt = restored_ckpt["opt"]
        ema_state = eqx.filter(restored_ckpt["ema_state"], eqx.is_array)
        steps = steps - resume_step
    logging.info(
        f"Number of parameters: {sum(param.size for param in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))}"
    )
    total_train_loss = 0
    total_steps = 0
    wandb.login(key=config.wandb_key)
    wandb.init(
        project="uot-fm",
        group=config.wandb_group,
        entity=config.wandb_entity,
        name=config.name,
        config=config,
    )
    for step in tqdm(range(steps), total=steps):
        train_key, resample_key = jr.split(train_key, 2)
        if config.task == "translation":
            src_batch, tgt_batch = next(train_src_loader), next(train_tgt_loader)
        else:
            sample_key, train_key = jr.split(train_key, 2)
            src_batch, tgt_batch = train_loader(sample_key)
        src_batch, tgt_batch = jnp.array(src_batch), jnp.array(tgt_batch)
        if config.training.matching:
            # resample batches
            src_batch, tgt_batch = batch_resampler(resample_key, src_batch, tgt_batch)
        # shard data
        src_batch, tgt_batch = jax.device_put([src_batch, tgt_batch], shard)
        # train step
        train_loss, model, train_key, opt_state = train_step_fn(
            model,
            tgt_batch,
            src_batch,
            train_key,
            opt_state,
        )
        if config.optim.ema_decay > 0.0:
            ema_state = update_ema(model, ema_state)
        total_train_loss += train_loss
        total_steps += 1
        if (step % config.training.print_freq) == 0 and step != 0 or step == steps - 1:
            # log train loss
            logging.info(f"Step {step}, Loss: {total_train_loss.item() / total_steps}")
            wandb.log({"train_loss": total_train_loss.item() / total_steps}, step=step)
            total_train_loss = 0
            total_steps = 0
        if (step % config.training.eval_freq) == 0 and step != 0 or step == steps - 1:
            if config.eval.compute_metrics:
                logging.info(f"Step {step}, Computing metrics...")
                if config.optim.ema_decay < 1.0:
                    combined_model = eqx.combine(ema_state.ema, static)
                else:
                    combined_model = model
                inference_model = eqx.tree_inference(combined_model, value=True)
                eval_key, metrics_key = jr.split(eval_key, 2)
                eval_dict = metric_computer.compute_metrics(inference_model, metrics_key)
                logging.info(f"Step {step}, Metrics: {eval_dict}")
                wandb.log(eval_dict, step=step)
                if config.training.save_checkpoints:
                    ckpt_mngr.save(
                        step,
                        combined_model,
                        metrics=eval_dict[config.eval.checkpoint_metric],
                    )
        if config.training.preemption_ckpt and step % config.training.ckpt_freq == 0:
            preemption_ckpt_mngr.save(
                step,
                {
                    "model": model,
                    "ema_model": ema_state,
                    "opt_state": opt_state,
                    "opt": opt,
                },
            )
    return model

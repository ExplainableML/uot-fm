from typing import Callable, List, Tuple
from diffusers import FlaxAutoencoderKL
import ml_collections
import jax
import jax.numpy as jnp
import jax.random as jr

from models.mlpmixer import Mixer2d
from models.unet import UNet


def get_model(config: ml_collections.ConfigDict, data_shape: List[int], model_key: jr.KeyArray):
    if config.model.type == "mlpmixer":
        return Mixer2d(
            data_shape,
            patch_size=config.model.patch_size,
            hidden_size=config.model.hidden_size,
            mix_patch_size=config.model.mix_patch_size,
            mix_hidden_size=config.model.mix_hidden_size,
            num_blocks=config.model.num_blocks,
            t1=config.t1,
            key=model_key,
        )
    elif config.model.type == "unet":
        return UNet(
            data_shape,
            is_biggan=config.model.biggan_sample,
            dim_mults=config.model.dim_mults,
            hidden_size=config.model.hidden_size,
            heads=config.model.heads,
            dim_head=config.model.dim_head,
            dropout_rate=config.model.dropout,
            num_res_blocks=config.model.num_res_blocks,
            attn_resolutions=config.model.attention_resolution,
            key=model_key,
        )
    else:
        raise ValueError(f"Unknown model type {config.model.type}")


def get_vae_fns(shard: jax.sharding.Sharding) -> Tuple[Callable, Callable]:
    fx_path = "CompVis/stable-diffusion-v1-4"
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(fx_path, subfolder="vae", revision="flax", dtype=jnp.float32)
    # replicate vae params across all devices
    vae_params = jax.device_put(vae_params, shard.replicate())

    @jax.jit
    def encode_fn(image_batch: jax.Array) -> jax.Array:
        latent_out = vae.apply({"params": vae_params}, image_batch, method=vae.encode)
        latent = latent_out.latent_dist.mode()
        latent = (latent * vae.config.scaling_factor).transpose(0, 3, 1, 2)
        return jax.lax.with_sharding_constraint(latent, shard)

    @jax.jit
    def decode_fn(latent_batch: jax.Array) -> jax.Array:
        image_out = vae.apply({"params": vae_params}, latent_batch / vae.config.scaling_factor, method=vae.decode)
        return jax.lax.with_sharding_constraint(image_out.sample, shard)

    return encode_fn, decode_fn

from typing import Callable

from absl import app
from absl import flags
import functools as ft
import einops
import jax
import jax.numpy as jnp
import logging
import numpy as np
import tensorflow as tf

tf.config.experimental.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds
from tqdm import tqdm

from models import inception
from utils.datasets import celeba_attribute, central_crop, cifar10, emnist

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 128, "Batch size for computing reference statistics.")


def prepare_dataset(data: np.ndarray, ds_name: str, batch_size: int) -> tfds.as_numpy:
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(
        get_preprocess_fn(ds_name),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = tfds.as_numpy(dataset)
    return dataset


def get_preprocess_fn(ds_name: str) -> Callable[[np.ndarray], tf.Tensor]:
    """Get preprocessing function for dataset."""

    def process_ds(x: np.ndarray):
        x = tf.cast(x, tf.float32) / 127.5 - 1.0
        if ds_name == "celeba256":
            x = tf.image.resize(x, [313, 256], antialias=True)
            x = central_crop(x, size=256)
        elif ds_name == "celeba64":
            x = tf.image.resize(x, [64, 64], antialias=True)
        elif ds_name == "emnist":
            return x
        return tf.transpose(x, perm=[2, 0, 1])

    return process_ds


def compute_fid_reference_stats(batch_size: int):
    logger = logging.getLogger()
    logger.setLevel("INFO")
    # load pretrained inceptionv3 model and setup jitted function
    rng = jax.random.PRNGKey(0)
    model = inception.InceptionV3(pretrained=True)
    params = model.init(rng, jnp.ones((1, 299, 299, 3)))
    apply_fn = ft.partial(model.apply, train=False)

    def compute_inception_acts(image_batch: jax.Array, repeat) -> jax.Array:
        inception_input = einops.repeat(image_batch, "b c h w -> b h w (c repeat)", repeat=repeat)
        inception_input = jax.image.resize(
            inception_input,
            shape=[image_batch.shape[0], 299, 299, 3],
            method="bilinear",
            antialias=True,
        )
        inception_output = apply_fn(params, jax.lax.stop_gradient(inception_input))
        return inception_output.squeeze(axis=1).squeeze(axis=1)

    # compute emnist stats
    logging.info("Computing reference statistics for emnist")
    _, dataset, _, _ = emnist("test")
    loader = prepare_dataset(dataset, "emnist", batch_size)
    inception_acts = []
    for batch in tqdm(loader):
        batch = jnp.array(batch)
        inception_acts.append(compute_inception_acts(batch, 3))
    inception_acts = jnp.concatenate(inception_acts, axis=0)
    mu = jnp.mean(inception_acts, axis=0)
    sigma = jnp.cov(inception_acts, rowvar=False)
    np.savez(f"assets/stats/emnist_letters.npz", mu=mu, sigma=sigma)
    _, dataset, _, labels = emnist("full")
    for label in [0, 1, 2]:
        logging.info(f"Computing reference statistics for emnist, label {label}")
        sub_dataset = dataset[np.array(labels[:, label], dtype=bool)]
        loader = prepare_dataset(sub_dataset, "emnist", batch_size)
        inception_acts = []
        for batch in tqdm(loader):
            batch = jnp.array(batch)
            inception_acts.append(compute_inception_acts(batch, 3))
        inception_acts = jnp.concatenate(inception_acts, axis=0)
        mu = jnp.mean(inception_acts, axis=0)
        sigma = jnp.cov(inception_acts, rowvar=False)
        np.savez(f"assets/stats/emnist_letters_{label}.npz", mu=mu, sigma=sigma)

    # compyte cifar10 stats
    logging.info("Computing reference statistics for cifar10")
    dataset = cifar10("train")
    loader = prepare_dataset(dataset, "cifar", batch_size)
    inception_acts = []
    for batch in tqdm(loader):
        batch = jnp.array(batch)
        inception_acts.append(compute_inception_acts(batch, 1))
    inception_acts = jnp.concatenate(inception_acts, axis=0)
    mu = jnp.mean(inception_acts, axis=0)
    sigma = jnp.cov(inception_acts, rowvar=False)
    np.savez(f"assets/stats/cifar10_train.npz", mu=mu, sigma=sigma)

    # compute celeba256 stats
    celeba_attribute_dict = {
        "male": {"attribute_id": 20, "map_forward": True, "subset_attributes": [15, 17, 35]},
        "female": {"attribute_id": 20, "map_forward": False, "subset_attributes": [15, 17, 35]},
        "add-glasses": {"attribute_id": 15, "map_forward": True, "subset_attributes": [17, 20, 201, 35]},
        "remove-glasses": {"attribute_id": 15, "map_forward": False, "subset_attributes": [17, 20, 201, 35]},
    }
    for name, data_args in celeba_attribute_dict.items():
        subset_attributes = data_args.pop("subset_attributes")
        logging.info(f"Computing reference statistics for celeba256 {name}")
        _, target_data, _, _ = celeba_attribute("test", batch_size=batch_size, overfit_to_one_batch=False, **data_args)
        loader = prepare_dataset(target_data, "celeba256", batch_size)
        inception_acts = []
        for batch in tqdm(loader):
            batch = jnp.array(batch)
            inception_acts.append(compute_inception_acts(batch, 1))
        inception_acts = jnp.concatenate(inception_acts, axis=0)
        mu = jnp.mean(inception_acts, axis=0)
        sigma = jnp.cov(inception_acts, rowvar=False)
        np.savez(f"assets/stats/celeba256_{name}.npz", mu=mu, sigma=sigma)
        logging.info(f"Computing reference statistics for celeba64 {name}")
        loader = prepare_dataset(target_data, "celeba64", batch_size)
        inception_acts = []
        for batch in tqdm(loader):
            batch = jnp.array(batch)
            inception_acts.append(compute_inception_acts(batch, 1))
        inception_acts = jnp.concatenate(inception_acts, axis=0)
        mu = jnp.mean(inception_acts, axis=0)
        sigma = jnp.cov(inception_acts, rowvar=False)
        np.savez(f"assets/stats/celeba64_{name}.npz", mu=mu, sigma=sigma)
        # compute celeba labelwise stats
        for subset_attribute in subset_attributes:
            logging.info(f"Computing reference statistics for celeba256 {name}, label {subset_attribute}")
            _, target_data, _, _ = celeba_attribute(
                "full",
                batch_size=batch_size,
                overfit_to_one_batch=False,
                subset_attribute_id=subset_attribute,
                **data_args,
            )
            loader = prepare_dataset(target_data, "celeba256", batch_size)
            inception_acts = []
            for batch in tqdm(loader):
                batch = jnp.array(batch)
                inception_acts.append(compute_inception_acts(batch, 1))
            inception_acts = jnp.concatenate(inception_acts, axis=0)
            mu = jnp.mean(inception_acts, axis=0)
            sigma = jnp.cov(inception_acts, rowvar=False)
            np.savez(f"assets/stats/celeba256_{name}_{subset_attribute}.npz", mu=mu, sigma=sigma)
            logging.info(f"Computing reference statistics for celeba64 {name}, label {subset_attribute}")
            loader = prepare_dataset(target_data, "celeba64", batch_size)
            inception_acts = []
            for batch in tqdm(loader):
                inception_acts.append(compute_inception_acts(batch, 1))
            inception_acts = jnp.concatenate(inception_acts, axis=0)
            mu = jnp.mean(inception_acts, axis=0)
            sigma = jnp.cov(inception_acts, rowvar=False)
            np.savez(f"assets/stats/celeba64_{name}_{subset_attribute}.npz", mu=mu, sigma=sigma)


if __name__ == "__main__":
    app.run(compute_fid_reference_stats)

from typing import Callable, List, Optional, Tuple

import csv
import jax
import jax.numpy as jnp
import logging
from ml_collections import ConfigDict
import numpy as np
import os
import cv2
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

from utils import GenerationSampler


def get_translation_datasets(
    config: ConfigDict,
    shard: Optional[jax.sharding.Sharding] = None,
    vae_encode_fn: Optional[Callable] = None,
) -> list[tf.data.Dataset]:
    """Get translation datasets and prepare them."""
    train_source_data, train_target_data, eval_source_data, eval_target_data = get_data(config, shard, vae_encode_fn)
    train_source_ds = prepare_dataset(train_source_data, config)
    eval_source_ds = prepare_dataset(eval_source_data, config, evaluation=True)
    train_target_ds = prepare_dataset(train_target_data, config)
    eval_target_ds = prepare_dataset(eval_target_data, config, evaluation=True)
    return train_source_ds, train_target_ds, eval_source_ds, eval_target_ds


def prepare_dataset(data: np.ndarray, config: ConfigDict, evaluation: bool = False) -> tfds.as_numpy:
    """Prepare dataset given config."""
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(
        get_preprocess_fn(config, evaluation),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    if not evaluation:
        dataset = dataset.shuffle(config.data.shuffle_buffer)
        dataset = dataset.repeat()
    dataset = dataset.batch(config.training.batch_size, drop_remainder=not evaluation)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = tfds.as_numpy(dataset)
    if len(data) == 2:
        dataset.length = data[0].shape[0]
    else:
        dataset.length = data.shape[0]
    return dataset


def get_preprocess_fn(config, evaluation: bool = False, precomputing: bool = False):
    """Get preprocessing function for dataset."""

    def process_ds(x: np.ndarray) -> tf.Tensor:
        x = tf.cast(x, tf.float32) / 127.5 - 1.0
        if config.data.source == "celeba_attribute":
            x = tf.image.resize(x, config.data.shape[1:], antialias=True)
            if config.data.random_crop:
                if not evaluation and not config.overfit_to_one_batch:
                    x = tf.image.random_crop(x, size=config.data.crop_shape)
                else:
                    x = central_crop(x, size=config.data.crop_shape[0])
            x = tf.transpose(x, perm=[2, 0, 1])
        elif config.task == "generation":
            x = tf.image.random_flip_left_right(x)
            x = tf.transpose(x, perm=[2, 0, 1])
        return x

    if config.model.use_vae and not precomputing:
        process_ds = lambda x: tf.cast(x, tf.float32)
    if evaluation and not precomputing:
        return lambda x, y: (process_ds(x), y)
    return process_ds


def central_crop(image: tf.Tensor, size: int) -> tf.Tensor:
    """Crop the center of an image to the given size."""
    top = (image.shape[0] - size) // 2
    left = (image.shape[1] - size) // 2
    return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_data(
    config: ConfigDict, shard: Optional[jax.sharding.Sharding] = None, vae_encode_fn: Optional[Callable] = None
) -> list[np.ndarray]:
    """Load source and target, train and evaluation data."""
    if config.data.target == "emnist":
        (
            train_source_data,
            train_target_data,
            train_source_label,
            train_target_label,
        ) = emnist("train")
        (
            eval_source_data,
            eval_target_data,
            eval_source_label,
            eval_target_label,
        ) = emnist("test")
    elif config.data.target == "celeba_attribute":
        if vae_encode_fn is not None:
            preprocess_fn = get_preprocess_fn(config, evaluation=True, precomputing=True)
        else:
            preprocess_fn = None
        (
            train_source_data,
            train_target_data,
            train_source_label,
            train_target_label,
        ) = celeba_attribute(
            "train",
            config.data.attribute_id,
            config.data.map_forward,
            config.training.batch_size,
            config.overfit_to_one_batch,
            shard,
            vae_encode_fn,
            preprocess_fn,
        )
        (
            eval_source_data,
            eval_target_data,
            eval_source_label,
            eval_target_label,
        ) = celeba_attribute(
            "test",
            config.data.attribute_id,
            config.data.map_forward,
            config.training.batch_size,
            config.overfit_to_one_batch,
            shard,
            vae_encode_fn,
            preprocess_fn,
        )
    elif config.data.target == "gaussian":
        train_source_data, train_target_data = get_unbalanced_uniform_samplers(
            input_dim=config.input_dim,
            num_samples=config.num_samples,
        )
        eval_source_data, eval_target_data = get_unbalanced_uniform_samplers(
            input_dim=config.input_dim,
            num_samples=config.eval.eval_samples,
        )
        eval_source_label, eval_target_label = None, None
    else:
        raise ValueError(f"Unknown target dataset {config.target_dataset}")

    # for translation between different datasets
    if config.data.source == "gaussian":
        pass
    elif config.data.source == "celeba_attribute":
        pass
    elif config.data.source == "emnist":
        pass
    else:
        raise ValueError(f"Unknown source dataset {config.data.source}")

    if config.overfit_to_one_batch:
        train_source_data = eval_source_data[: config.training.batch_size]
        train_target_data = eval_target_data[: config.training.batch_size]
        eval_source_data = train_source_data
        eval_target_data = train_target_data
        eval_source_label = eval_source_label[: config.training.batch_size]
        eval_target_label = eval_target_label[: config.training.batch_size]
    return (
        train_source_data,
        train_target_data,
        (eval_source_data, eval_source_label),
        (eval_target_data, eval_target_label),
    )


def emnist(split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load emnist data from numpy files."""
    target_dir = os.getcwd() + "/data/emnist"
    if split == "train":
        data_x = np.load(f"{target_dir}/x_train.npy")
        data_y = np.load(f"{target_dir}/y_train.npy")
    elif split == "test":
        data_x = np.load(f"{target_dir}/x_test.npy")
        data_y = np.load(f"{target_dir}/y_test.npy")
    elif split == "full":
        data_x = np.concatenate([np.load(f"{target_dir}/x_train.npy"), np.load(f"{target_dir}/x_test.npy")])
        data_y = np.concatenate([np.load(f"{target_dir}/y_train.npy"), np.load(f"{target_dir}/y_test.npy")])
    digits_indices = np.isin(data_y, np.array([0, 1, 8]))
    letters_indices = np.logical_not(digits_indices)
    source_data = data_x[digits_indices]
    target_data = data_x[letters_indices]
    # Map labels to 0, 1, 2
    map_fn = np.vectorize({0: 0, 1: 1, 8: 2, 11: 2, 18: 1, 24: 0}.__getitem__)
    data_y = map_fn(data_y)
    source_label = data_y[digits_indices]
    target_label = data_y[letters_indices]
    one_hot_src_labels = np.eye(3)[source_label]
    one_hot_tgt_labels = np.eye(3)[target_label]
    return source_data, target_data, one_hot_src_labels, one_hot_tgt_labels


def celeba_attribute(
    split: str,
    attribute_id: int,
    map_forward: bool,
    batch_size: int,
    overfit_to_one_batch: bool,
    shard: Optional[jax.sharding.Sharding] = None,
    vae_encode_fn: Optional[Callable] = None,
    preprocess_fn: Optional[Callable] = None,
    subset_attribute_id: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load celeba attribute data.

    Args:
        split: Train, test or full split
        attribute_id: Attribute id to split on (0-39)
        map_forward: Whether to map forward or backward
        batch_size: Batch size
        overfit_to_one_batch: Whether to overfit to one batch
        shard: Sharding object for vae encoding
        vae_encode_fn: Vae encode function
        preprocess_fn: Preprocess function
        subset_attribute_id: Subset attribute id to split on (0-39)
    """

    data_dir = "./data/celeba"
    with open(f"{data_dir}/list_attr_celeba.txt") as csv_file:
        data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))
        data = data[2:]
        filenames = [row[0] for row in data]
        data = [row[1:] for row in data]
        label_int = np.array([list(map(int, i)) for i in data])

    with open(f"{data_dir}/list_eval_partition.txt") as csv_file:
        data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))
        data = [row[1:] for row in data]
        split_int = np.array([list(map(int, i)) for i in data])

    # get indices for split and attribute
    if split == "train":
        splits = 0
    elif split == "test":
        splits = [1, 2]
    elif split == "full":
        splits = [0, 1, 2]
    split_indices = np.isin(split_int, splits).squeeze()
    if map_forward:
        source_indices = label_int[:, attribute_id] != 1
        target_indices = label_int[:, attribute_id] == 1
    else:
        source_indices = label_int[:, attribute_id] == 1
        target_indices = label_int[:, attribute_id] != 1
    if subset_attribute_id is not None:
        if subset_attribute_id == 201:
            # subset for glasses
            source_indices = source_indices * (label_int[:, 20] != 1)
            target_indices = target_indices * (label_int[:, 20] != 1)
        else:
            source_indices = source_indices * (label_int[:, subset_attribute_id] == 1)
            target_indices = target_indices * (label_int[:, subset_attribute_id] == 1)

    # get filenames
    source_indices = split_indices * source_indices
    target_indices = split_indices * target_indices
    source_filenames = [filename for filename, indice in zip(filenames, source_indices) if indice]
    source_labels = np.array([label for label, indice in zip(label_int, source_indices) if indice])
    target_filenames = [filename for filename, indice in zip(filenames, target_indices) if indice]
    target_labels = np.array([label for label, indice in zip(label_int, target_indices) if indice])

    logging.info("Loading source and target data.")
    source_data = []
    target_data = []
    for fname in tqdm(source_filenames):
        image = cv2.imread(f"{data_dir}/img_align_celeba/{fname}")
        # cv2 reads images in BGR format, so we need to reverse the channel
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if vae_encode_fn is not None:
            image = preprocess_fn(image).numpy()
        source_data.append(image)
        if overfit_to_one_batch and len(source_data) == batch_size:
            break

    for fname in tqdm(target_filenames):
        image = cv2.imread(f"{data_dir}/img_align_celeba/{fname}")
        # cv2 reads images in BGR format, so we need to reverse the channel
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if vae_encode_fn is not None:
            image = preprocess_fn(image).numpy()
        target_data.append(image)
        if overfit_to_one_batch and len(target_data) == batch_size:
            break
    if vae_encode_fn is not None:
        logging.info("Precomputing VAE embedding.")
        batch_size = batch_size // 2
        vae_source_data = []
        vae_target_data = []
        # compute vae embedding batch-wise
        for idx in tqdm(range(0, len(source_data), batch_size)):
            batch = np.array(source_data[idx : idx + batch_size])
            num_pad = batch_size - batch.shape[0]
            if batch.shape[0] < batch_size:
                # pad batch, shard and then unpad again
                batch = np.concatenate([batch, np.zeros([num_pad, *batch.shape[1:]])])
            batch = jax.device_put(batch, shard)
            vae_out = vae_encode_fn(batch)
            if num_pad > 0:
                vae_out = vae_out[:-num_pad]
            vae_source_data.append(vae_out)
        for idx in tqdm(range(0, len(target_data), batch_size)):
            batch = np.array(target_data[idx : idx + batch_size])
            num_pad = batch_size - batch.shape[0]
            if batch.shape[0] < batch_size:
                # pad batch, shard and then unpad again
                batch = np.concatenate([batch, np.zeros([num_pad, *batch.shape[1:]])])
            batch = jax.device_put(batch, shard)
            vae_out = vae_encode_fn(batch)
            if num_pad > 0:
                vae_out = vae_out[:-num_pad]
            vae_target_data.append(vae_out)
        source_data = np.concatenate(vae_source_data)
        target_data = np.concatenate(vae_target_data)
    else:
        source_data = np.array(source_data)
        target_data = np.array(target_data)
    return source_data, target_data, source_labels, target_labels


def get_generation_datasets(config: ConfigDict) -> GenerationSampler:
    """Get generation dataset and create sampler."""
    train_data = cifar10("train")
    return GenerationSampler(jnp.array(train_data), config.training.batch_size)


def cifar10(split: str) -> np.ndarray:
    """Load cifar10 data from tensorflow datasets."""
    [x_train, y_train], [x_test, y_test] = tf.keras.datasets.cifar10.load_data()
    if split == "train":
        return x_train
    else:
        return x_test


def get_unbalanced_uniform_samplers(
    input_dim: int = 2,
    num_samples: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate unbalanced Gaussian data and return a tuple of data samplers."""
    # generate source data
    source_center_one = np.repeat(np.array([0, -1])[None, :], int(num_samples * 1.5), axis=0)
    source_center_two = np.repeat(np.array([5, -1])[None, :], num_samples, axis=0)
    source_center = np.concatenate([source_center_one, source_center_two])
    source_data = source_center + np.random.uniform(
        size=[int(num_samples * 1.5) + num_samples, input_dim], low=-0.5, high=0.5
    )
    # generate target data
    target_center_one = np.repeat(np.array([0, 1])[None, :], num_samples, axis=0)
    target_center_two = np.repeat(np.array([5, 1])[None, :], int(num_samples * 1.5), axis=0)
    target_center = np.concatenate([target_center_one, target_center_two])
    target_data = target_center + np.random.uniform(
        size=[int(num_samples * 1.5) + num_samples, input_dim], low=-0.5, high=0.5
    )
    return source_data, target_data

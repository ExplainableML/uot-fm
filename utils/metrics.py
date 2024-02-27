from typing import Callable, Optional

import functools as ft
import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import ml_collections
import numpy as np
import scipy
import tensorflow as tf
from tqdm import tqdm
import wandb
import warnings

from models import inception


class MetricComputer:
    """
    Class to compute metrics for evaluation.
    """

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        shard: jax.sharding.Sharding,
        eval_ds: tf.data.Dataset,
        sample_fn: Callable,
        vae_decode_fn: Optional[Callable] = None,
        vae_encode_fn: Optional[Callable] = None,
    ):
        # load pretrained inceptionv3 model
        rng = jax.random.PRNGKey(0)
        model = inception.InceptionV3(pretrained=True)
        self.params = model.init(rng, jnp.ones((1, 299, 299, 3)))
        self.apply_fn = jax.jit(ft.partial(model.apply, train=False))
        # get training config
        self.eval_labelwise = config.eval.labelwise
        self.task = config.task
        if self.task == "translation":
            self.num_eval_samples = eval_ds.length
        else:
            self.num_eval_samples = config.eval.eval_samples
        self.batch_size = config.training.batch_size
        self.return_samples = config.eval.save_samples
        if self.return_samples:
            self.num_save_samples = config.eval.num_save_samples
        self.dataset = eval_ds
        self.repeat = 3 if config.data.shape[0] == 1 else 1
        self.input_shape = config.model.input_shape
        self.sample_fn = sample_fn
        self.enable_fid = config.eval.enable_fid
        self.enable_mse = config.eval.enable_mse
        self.enable_path_lengths = config.eval.enable_path_lengths
        if self.enable_mse:
            self.mse_fn = jax.jit(lambda x, y: jnp.mean((x - y) ** 2))
        if self.enable_path_lengths:
            self.rmse_fn = jax.jit(lambda x, y: jnp.mean(jnp.sqrt((x - y) ** 2)))
        self.shard = shard
        self.use_vae = config.model.use_vae
        if self.use_vae:
            assert vae_decode_fn is not None and vae_encode_fn is not None
            self.vae_decode_fn = vae_decode_fn
            self.vae_encode_fn = vae_encode_fn
        if self.enable_fid:
            # get statistics for real data
            stats_file_name = config.data.precomputed_stats_file
            precomputed_stats = np.load(f"./assets/stats/{stats_file_name}.npz")
            self.mu_real, self.sigma_real = (
                precomputed_stats["mu"],
                precomputed_stats["sigma"],
            )
            if self.eval_labelwise:
                self.eval_labels = jnp.array(config.data.eval_labels)
                self.mus_real, self.sigmas_real = [], []
                for label in self.eval_labels:
                    precomputed_stats = np.load(f"./assets/stats/{stats_file_name}_{label}.npz")
                    self.mus_real.append(precomputed_stats["mu"])
                    self.sigmas_real.append(precomputed_stats["sigma"])

    def compute_metrics(self, model: eqx.Module, key: jr.KeyArray):
        """
        Compute metrics for evaluation.

        Args:
            model: model to evaluate
            key: jax random key

        Returns:
            eval_dict: dictionary with evaluation metrics and samples for wandb logging
        """
        eval_dict = {}
        if self.eval_labelwise:
            labels_indices = [[] for _ in range(self.eval_labels.shape[0])]
        samples = None
        inputs = None
        mses = []
        path_lengths = []
        inception_acts = []
        nfes = []
        # create vmap functions
        partial_sample_fn = ft.partial(self.sample_fn, model)
        # compute metrics batch-wise
        eval_num_iter = self.num_eval_samples // self.batch_size
        if self.task == "translation":
            loader = iter(self.dataset)
        for _ in tqdm(range(eval_num_iter)):
            if self.task == "translation":
                batch = next(loader)
                # get data
                if self.eval_labelwise:
                    src_batch, src_label = batch
                else:
                    src_batch = batch
                # padding for last batch if necessary
                pad_size = self.batch_size - src_batch.shape[0]
                if pad_size > 0:
                    # pad batch, shard and unpad
                    src_batch = jnp.pad(src_batch, ((0, pad_size), (0, 0), (0, 0), (0, 0)))
            else:
                sample_key, key = jr.split(key, 2)
                pad_size = 0
                src_batch = jr.normal(sample_key, [self.batch_size, *self.input_shape])
            src_batch = jax.device_put(src_batch, self.shard)
            if inputs is None:
                if self.use_vae:
                    inputs = self.vae_decode_fn(src_batch) * 0.5 + 0.5
                else:
                    inputs = src_batch * 0.5 + 0.5
            elif inputs.shape[0] < 2400:
                if self.use_vae:
                    inputs = jnp.concatenate(
                        [inputs, self.vae_decode_fn(src_batch)[: int(2400 - inputs.shape[0])] * 0.5 + 0.5]
                    )
                else:
                    inputs = jnp.concatenate([inputs, src_batch[: int(2400 - inputs.shape[0])] * 0.5 + 0.5])
            # sample from model
            sample_batch, nfe = jax.vmap(partial_sample_fn)(src_batch)
            nfes.append(nfe)
            if self.enable_path_lengths:
                # compute euclidean distance between samples and inputs
                if pad_size > 0:
                    path_lengths.append(self.rmse_fn(src_batch[:-pad_size], sample_batch[:-pad_size]))
                else:
                    path_lengths.append(self.rmse_fn(src_batch, sample_batch))
            if self.use_vae:
                sample_batch = jax.device_put(sample_batch, self.shard)
                sample_batch = self.vae_decode_fn(sample_batch)
            sample_batch = jnp.clip(sample_batch, -1.0, 1.0)
            if self.enable_fid:
                inception_act = self.compute_inception_acts(sample_batch)
                inception_acts.append(inception_act)
            if pad_size > 0:
                src_batch = src_batch[:-pad_size]
                sample_batch = sample_batch[:-pad_size]
            # safe samples and compute inception activation
            if samples is None:
                samples = sample_batch * 0.5 + 0.5
            elif samples.shape[0] < 2400:
                samples = jnp.concatenate([samples, sample_batch[: int(2400 - samples.shape[0])] * 0.5 + 0.5])
            if self.eval_labelwise:
                for idx, label in enumerate(self.eval_labels):
                    if label == 201:
                        labels_indices[idx].append((src_label[:, 20] == -1))
                    else:
                        labels_indices[idx].append((src_label[:, label] == 1.0))
        eval_dict["nfe"] = jnp.mean(jnp.hstack(nfes))
        if self.enable_mse:
            eval_dict["mse"] = jnp.mean(jnp.hstack(mses))
        if self.enable_path_lengths:
            eval_dict["path_lengths"] = jnp.mean(jnp.hstack(path_lengths)) * 127.5
            eval_dict["path_lengths_std"] = jnp.std(jnp.hstack(path_lengths)) * 127.5
        # compute fid
        if self.enable_fid:
            inception_acts = jnp.concatenate(inception_acts, axis=0)
            if pad_size > 0:
                inception_acts = inception_acts[:-pad_size]
            mu = jnp.mean(inception_acts, axis=0)
            sigma = jnp.cov(inception_acts, rowvar=False)
            eval_dict["fid"] = self.compute_fid(self.mu_real, self.sigma_real, mu, sigma)
        if self.return_samples:
            # save image grid loggable to wandb
            if self.task == "generation":
                image_grid = jnp.concatenate([samples[: self.num_save_samples**2]])
                image_grid = einops.rearrange(
                    image_grid,
                    "(n m) c h w -> (n h) (m w) c",
                    n=self.num_save_samples,
                    m=self.num_save_samples,
                )
            else:
                nmb_double_rows = self.num_save_samples // 2
                # create image grid of alternating rows of input and output
                rows = []
                for row_idx in range(nmb_double_rows):
                    rows.append(inputs[row_idx * self.num_save_samples : (row_idx + 1) * self.num_save_samples])
                    rows.append(samples[row_idx * self.num_save_samples : (row_idx + 1) * self.num_save_samples])
                image_grid = jnp.concatenate(rows)
                image_grid = einops.rearrange(
                    image_grid,
                    "(n m) c h w -> (n h) (m w) c",
                    n=nmb_double_rows * 2,
                    m=self.num_save_samples,
                )
            eval_dict["samples"] = wandb.Image(np.array(image_grid))
            if self.eval_labelwise:
                # compute fid labelwise
                fid_scores = []
                for idx, label in enumerate(self.eval_labels):
                    label_indices = jnp.concatenate(labels_indices[idx], axis=0)
                    if self.enable_fid:
                        inception_act_label = inception_acts[label_indices]
                        mu = jnp.mean(inception_act_label, axis=0)
                        sigma = jnp.cov(inception_act_label, rowvar=False)
                        fid_score = self.compute_fid(self.mus_real[idx], self.sigmas_real[idx], mu, sigma)
                        eval_dict[f"fid_{label}"] = fid_score
                        fid_scores.append(fid_score)
                    if self.return_samples:
                        # save image grid loggable to wandb
                        plot_indices = label_indices[:2400]
                        rows = []
                        for row_idx in range(nmb_double_rows):
                            rows.append(
                                inputs[plot_indices][
                                    row_idx * self.num_save_samples : (row_idx + 1) * self.num_save_samples
                                ]
                            )
                            rows.append(
                                samples[plot_indices][
                                    row_idx * self.num_save_samples : (row_idx + 1) * self.num_save_samples
                                ]
                            )
                        image_grid = jnp.concatenate(rows)
                        image_grid = einops.rearrange(
                            image_grid,
                            "(n m) c h w -> (n h) (m w) c",
                            n=nmb_double_rows * 2,
                            m=self.num_save_samples,
                        )
                        eval_dict[f"samples_{label}"] = wandb.Image(np.array(image_grid))
                eval_dict["fid_average"] = jnp.mean(jnp.hstack(fid_scores))
        return eval_dict

    def compute_inception_acts(self, image_batch: jax.Array) -> jax.Array:
        """
        Compute inception activations for a batch of images.
        """
        inception_input = einops.repeat(image_batch, "b c h w -> b h w (c repeat)", repeat=self.repeat)
        inception_input = jax.image.resize(
            inception_input,
            shape=[image_batch.shape[0], 299, 299, 3],
            method="bilinear",
            antialias=True,
        )
        inception_output = self.apply_fn(self.params, jax.lax.stop_gradient(inception_input))
        return inception_output.squeeze(axis=1).squeeze(axis=1)

    @staticmethod
    def compute_fid(
        mu_real: np.ndarray, sigma_real: np.ndarray, mu_gen: np.ndarray, sigma_gen: np.ndarray, eps: float = 1e-6
    ) -> np.ndarray:
        """
        Compute Frechet Inception Distance (FID) between two distributions.
        """
        # compute statistics
        mu_gen = np.atleast_1d(mu_gen)
        mu_real = np.atleast_1d(mu_real)
        sigma_gen = np.atleast_1d(sigma_gen)
        sigma_real = np.atleast_1d(sigma_real)

        assert mu_gen.shape == mu_real.shape
        assert sigma_gen.shape == sigma_real.shape

        diff = mu_real - mu_gen
        covmean, _ = scipy.linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)

        if not np.isfinite(covmean).all():
            warnings.warn((f"fid calculation produces singular product; " "adding {eps} to diagonal of cov estimates"))
            offset = np.eye(sigma_real.shape[0]) * eps
            covmean = scipy.linalg.sqrtm((sigma_real + offset).dot(sigma_gen + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-2):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * tr_covmean

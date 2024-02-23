from dataclasses import dataclass
from typing import Optional, Tuple

import equinox as eqx
import dm_pix as pix
import jax
import jax.numpy as jnp
import jax.random as jr
from ott.geometry.pointcloud import PointCloud
from ott.solvers.linear import sinkhorn


@dataclass
class GenerationSampler:
    """Data sampler for a generation task with optional weighting."""

    data: jax.Array
    batch_size: int
    weighting: Optional[jnp.ndarray] = None
    do_flip: bool = True

    def __post_init__(self):
        # Weighting needs to have the same length as data.
        if self.weighting is not None:
            assert self.data.shape[0] == self.weighting.shape[0]

        @eqx.filter_jit
        def _sample(key: jax.random.KeyArray) -> jax.Array:
            """Jitted sample function."""
            x = jax.random.choice(key, self.data, shape=[self.batch_size], p=self.weighting)
            x = x / 127.5 - 1.0
            if self.do_flip:
                x = pix.random_flip_left_right(key, x)
            x = jnp.transpose(x, [0, 3, 1, 2])
            return jr.normal(key, shape=x.shape), x

        self.sample = _sample

    def __call__(self, key: jax.random.KeyArray) -> jnp.ndarray:
        """Sample data."""
        return self.sample(key)


@dataclass
class BatchResampler:
    """Batch resampler based on (Unbalanced) Optimal Transport."""

    batch_size: int
    tau_a: float = 1.0
    tau_b: float = 1.0
    epsilon: float = 1e-2

    def __post_init__(self):
        @eqx.filter_jit(donate="all")
        def _resample(
            key: jr.KeyArray,
            source_batch: jax.Array,
            target_batch: jax.Array,
            source_labels: Optional[jax.Array] = None,
            target_labels: Optional[jax.Array] = None,
        ) -> Tuple[jax.Array, jax.Array]:
            """Jitted resample function."""
            # solve regularized ot between batch_source and batch_target reshaped to (batch_size, dimension)
            geom = PointCloud(
                jnp.reshape(source_batch, [self.batch_size, -1]),
                jnp.reshape(target_batch, [self.batch_size, -1]),
                epsilon=self.epsilon,
                scale_cost="mean",
            )
            ot_out = sinkhorn.solve(geom, tau_a=self.tau_a, tau_b=self.tau_b)

            # get flattened log transition matrix
            transition_matrix = jnp.log(ot_out.matrix.flatten())
            # sample from transition_matrix
            indeces = jax.random.categorical(key, transition_matrix, shape=[self.batch_size])
            resampled_indeces_source = indeces // self.batch_size
            resampled_indeces_target = indeces % self.batch_size
            if source_labels is None:
                return source_batch[resampled_indeces_source], target_batch[resampled_indeces_target]
            return (
                source_batch[resampled_indeces_source],
                target_batch[resampled_indeces_target],
                source_labels[resampled_indeces_source],
                target_labels[resampled_indeces_target],
            )

        self.resample = _resample

    def __call__(
        self,
        key: jr.KeyArray,
        source_batch: jax.Array,
        target_batch: jax.Array,
        source_labels: Optional[jax.Array] = None,
        target_labels: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """Sample data."""
        return self.resample(key, source_batch, target_batch, source_labels, target_labels)

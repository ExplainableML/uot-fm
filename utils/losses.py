from typing import Any, Callable, Dict, List, Optional, Tuple

import diffrax as dfx
import equinox as eqx
import functools as ft
import jax
import jax.numpy as jnp
import jax.random as jr
from ml_collections import ConfigDict
import optax


def get_loss_builder(config: ConfigDict):
    if config.training.method == "flow":
        return FlowMatching(
            t1=config.t1,
            dt0=config.dt0,
            flow_sigma=config.training.flow_sigma,
            gamma=config.training.gamma,
            weight=lambda t: 1.0,
            solver=config.solver,
        )
    elif config.training.method == "flow-vp-matching":
        raise NotImplementedError
    elif config.training.method == "flow-ve-matching":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown training method {config.training.method}")


def get_optimizer(config: ConfigDict):
    if config.optim.schedule == "constant":
        schedule = optax.constant_schedule(config.optim.learning_rate)
    elif config.optim.schedule == "linear":
        schedule = optax.linear_schedule(
            init_value=config.optim.learning_rate,
            end_value=1e-8,
            transition_steps=config.training.num_steps - config.optim.warmup,
        )
    elif config.optim.schedule == "polynomial":
        schedule = optax.polynomial_schedule(
            init_value=config.optim.learning_rate,
            end_value=1e-8,
            power=0.9,
            transition_steps=config.training.num_steps - config.optim.warmup,
        )
    elif config.optim.schedule == "cosine":
        schedule = optax.cosine_decay_schedule(
            init_value=config.optim.learning_rate,
            decay_steps=config.training.num_steps - config.optim.warmup,
            alpha=1e-5 / config.optim.learning_rate,
        )
    else:
        raise ValueError(f"Unknown schedule type {config.optim.schedule}")
    if config.optim.warmup > 0:
        warmup_schedule = optax.linear_schedule(
            init_value=1e-8,
            end_value=config.optim.learning_rate,
            transition_steps=config.optim.warmup,
        )
        schedule = optax.join_schedules(
            schedules=[warmup_schedule, schedule],
            boundaries=[config.optim.warmup],
        )
    if config.optim.optimizer == "adam":
        optimizer = optax.adamw(
            learning_rate=schedule,
            b1=config.optim.beta_one,
            b2=config.optim.beta_two,
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )
    elif config.optim.optimizer == "sgd":
        optimizer = optax.sgd(
            learning_rate=schedule,
            momentum=config.optim.momentum,
            nesterov=config.optim.nesterov,
        )
    elif config.optim.optimizer == "adabelief":
        optimizer = optax.adabelief(
            learning_rate=schedule,
            b1=config.optim.beta_one,
            b2=config.optim.beta_two,
            eps=config.optim.eps,
        )
    else:
        raise ValueError(f"Unknown optimizer type {config.optim.type}")
    if config.optim.grad_clip > 0.0:
        optimizer = optax.chain(optimizer, optax.clip_by_global_norm(config.optim.grad_clip))
    return optimizer


class FlowMatching:
    """Class for Flow Matching loss computation and sampling."""

    def __init__(
        self,
        t1: float,
        dt0: float,
        t0: float = 0.0,
        gamma: str = "constant",
        flow_sigma: Optional[float] = 0.1,
        weight: Optional[Callable[[float], float]] = lambda t: 1.0,
        solver: str = "tsit5",
    ):
        self.t1 = t1
        self.t0 = t0
        self.dt0 = dt0
        self.gamma = gamma
        self.sigma = flow_sigma
        self.weight = weight
        self.solver = solver

    @staticmethod
    def compute_flow(x1: jax.Array, x0: jax.Array) -> jax.Array:
        return x1 - x0

    @staticmethod
    def compute_mu_t(x1: jax.Array, x0: jax.Array, t: float) -> jax.Array:
        return t * x1 + (1 - t) * x0

    def compute_gamma_t(self, t: float) -> jax.Array:
        if self.gamma == "bridge":
            return self.sigma * jnp.sqrt(t * (1 - t))
        elif self.gamma == "constant":
            return self.sigma
        else:
            raise ValueError(f"Unknown noise schedule {self.gamma}")

    def sample_xt(self, x1: jax.Array, x0: jax.Array, t: float, noise: jax.Array) -> jax.Array:
        mu_t = self.compute_mu_t(x1, x0, t)
        return mu_t + self.compute_gamma_t(t) * noise

    def get_batch_loss_fn(self):
        """Get single loss function."""

        def single_loss_fn(
            model: eqx.Module,
            x1: jax.Array,
            x0: jax.Array,
            t: float,
            key: jr.KeyArray,
        ) -> jax.Array:
            noise = jr.normal(key, x1.shape)
            u_t = self.compute_flow(x1, x0)
            x_t = self.sample_xt(x1, x0, t, noise)
            pred = model(t, x_t, key=key)
            return self.weight(t) * jnp.mean((pred - u_t) ** 2)

        def batch_loss_fn(
            model: eqx.Module,
            x1: jax.Array,
            x0: jax.Array,
            key: jr.KeyArray,
        ) -> jax.Array:
            batch_size = x1.shape[0]
            tkey, losskey = jr.split(key)
            losskey = jr.split(losskey, batch_size)
            # Low-discrepancy sampling over t to reduce variance
            t = jr.uniform(tkey, (batch_size,), minval=0, maxval=self.t1 / batch_size)
            t = t + (self.t1 / batch_size) * jnp.arange(batch_size)
            loss_fn = jax.vmap(ft.partial(single_loss_fn, model))
            return jnp.mean(loss_fn(x1, x0, t, losskey))

        return batch_loss_fn

    def get_train_step_fn(self, loss_fn: Callable, opt_update: optax.GradientTransformation):
        """Returns a callable train function."""
        grad_value_loss_fn = eqx.filter_value_and_grad(loss_fn)

        @eqx.filter_jit
        def step(
            model: eqx.Module,
            x1: jax.Array,
            x0: jax.Array,
            key: jr.KeyArray,
            opt_state: optax.OptState,
        ) -> Tuple[jax.Array, eqx.Module, jr.KeyArray, optax.OptState]:
            loss, grads = grad_value_loss_fn(model, x1, x0, key)
            updates, opt_state = opt_update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            key = jr.split(key, 1)[0]
            return loss, model, key, opt_state

        return step

    def get_sample_fn(self):
        """Get single sample function."""

        @eqx.filter_jit
        def single_sample_fn(model: eqx.Module, x0: jax.Array) -> jax.Array:
            """Produce single sample from the CNF by integrating forward."""

            def func(t, x, args):
                return model(t, x)

            term = dfx.ODETerm(func)
            if self.solver == "tsit5":
                solver = dfx.Tsit5()
            elif self.solver == "euler":
                solver = dfx.Euler()
            elif self.solver == "heun":
                solver = dfx.Heun()
            else:
                raise ValueError(f"Unknown solver {self.solver}")
            if self.dt0 == 0.0:
                stepsize_controller = dfx.PIDController(rtol=1e-5, atol=1e-5)
            else:
                stepsize_controller = dfx.ConstantStepSize()
            sol = dfx.diffeqsolve(
                term,
                solver,
                self.t0,
                self.t1,
                self.dt0,
                x0,
                stepsize_controller=stepsize_controller,
            )
            return sol.ys[0], sol.stats["num_steps"]

        return single_sample_fn

import jax.numpy as jnp
import numpy as np
import structlog
from flax import nnx
from tqdm import trange
from typing import Any

from .data import Data

log = structlog.get_logger()


def train_step(
    model: nnx.Module, optimizer: nnx.Optimizer, x: jnp.ndarray, y: jnp.ndarray
) -> jnp.ndarray:
    """
    Performs a single training step using autodiff from nnx.
    Returns the scalar loss.
    """
    log.debug("Entering train_step", x_shape=x.shape, y_shape=y.shape)

    def loss_fn(model: nnx.Module):
        y_hat = model(x)
        return 0.5 * jnp.mean((y_hat - y) ** 2)
        """
        My basis functions are a little close together indicating redundancy and makes me afraid im overfitting.
        I want to introduce some sort of repulsive factor but I don't know how and I kind of give up on trying.
        """

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def train(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    data: Data,
    settings: Any,
    np_rng: np.random.Generator,
) -> None:
    """
    Train the provided model in-place using stochastic gradient descent.
    """
    log.info(
        "Starting training",
        iters=settings.num_iters,
        batch_size=settings.batch_size,
        lr=settings.learning_rate,
    )
    bar = trange(settings.num_iters)
    for i in bar:
        x_np, y_np = data.get_batch(np_rng, settings.batch_size)
        x = jnp.asarray(x_np)
        y = jnp.asarray(y_np)
        loss = train_step(model, optimizer, x, y)
        bar.set_description(f"Loss @ {i} => {float(loss):.6f}")
        bar.refresh()
        if (i + 1) % max(1, settings.num_iters // 5) == 0:
            log.info("Intermediate training status", iter=i + 1, loss=float(loss))
    log.info("Training finished")

import jax.numpy as jnp
import numpy as np
import optax
import structlog

from flax import nnx
from tqdm import trange
from .config import TrainingSettings
from .data import MNIST
from .model import Classify

log = structlog.get_logger() 

@nnx.jit
def train_step_mnist(
    model: Classify,
    optimizer: nnx.Optimizer,
    x: jnp.ndarray,
    y: jnp.ndarray,
 ):
    
    """perform on training step"""

    def loss_function(model: Classify):
        logits = model(x, True)
        ce_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
        L2_loss = model.L2_loss()

        return ce_loss + L2_loss
    
    loss, grads = nnx.value_and_grad(loss_function)(model)
    optimizer.update(model, grads)
    return loss

def MNIST_train(
    model: Classify,
    optimizer: nnx.Optimizer,
    data: MNIST,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
 ) -> None:
    """train with grad descend"""
    log.info("started training")
    bar = trange(settings.num_iters)
    for i in bar:
        x_np, y_np = data.get_batch(np_rng, settings.batch_size)
        log.debug("y_np", y_np=y_np)
        x, y = jnp.asarray(x_np), jnp.asarray(y_np)
        log.debug("y here", y=y)
        # x is (batch_size, 28, 28, 1)
        loss = train_step_mnist(model, optimizer, x, y)
        bar.set_description(f"loss at {i} => {loss:.6f}")
        bar.refresh()
    log.info("done training")
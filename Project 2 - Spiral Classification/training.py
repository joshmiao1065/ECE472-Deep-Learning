import jax.numpy as jnp
import numpy as np
import structlog

from flax import nnx
from tqdm import trange
from .config import TrainingSettings
from .data import Data
from .model import MLP

log = structlog.get_logger()

@nnx.jit
def train_step_spiral(
    model: MLP,
    optimizer: nnx.Optimizer,
    x: jnp.ndarray,
    target: jnp.ndarray,
    epsilon: float,
 ):

    def loss_function(model: MLP):
        """trianing step"""
        probs = model(x)
        #limit the max and min value of value passed into logarithm to avoid nan loss
        probs = jnp.where(probs < epsilon, epsilon, probs)
        probs = jnp.where(probs > 1 - epsilon, epsilon, probs)

        #calculate average loss using binary cross entropy
        loss = -((jnp.log(probs) * target) + ((1 - target) * jnp.log(1 - probs)))        
        loss = jnp.mean(loss)

        return loss
    
    loss, grads = nnx.value_and_grad(loss_function)(model)

    optimizer.update(model, grads)
    return loss

def spiral_train(
    model: MLP,
    optimizer: nnx.Optimizer,
    data: Data,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
):
    """SGD training"""
    log.info("Start training")
    bar = trange(settings.num_iters)
    for i in bar:
        x_np, target_np = data.get_batch(np_rng, settings.batch_size)
        x, target = jnp.asarray(x_np), jnp.asarray(target_np)
        target = target.reshape(-1, 1)  #reshape target to correct output diemnsions

        loss = train_step_spiral(model, optimizer, x, target, settings.epsilon)
        
        bar.set_description(f"Loss at {i} => {loss:.6f}")
        
        bar.refresh()
    log.info("Training Finished")
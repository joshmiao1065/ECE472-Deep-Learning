import jax
import numpy as np
import structlog
import optax

from flax import nnx
from numpy.random import Generator, PCG64

from .config import load_settings
from .data import Data
from .model import MLP
from .logging import configure_logging
from .plotting import spiral_plot
from .training import spiral_train

def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(472) #472 for ECE 472
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.array(data_key))

    data = Data(
        rng = Generator(PCG64(seed=472)),
        num_points = settings.data.num_samples // 2,
        num_turns = 4,
        scale = 1.0,
        noise = settings.data.noise,
    )
    log.info("spirals generated", shape = data.x.shape)

    model = MLP(
        rngs=nnx.Rngs(params=model_key), 
        num_inputs = settings.data.num_features,
        num_outputs = settings.data.num_outputs,
        num_hl=settings.model.num_hl,
        hl_width=settings.model.hl_width,
    )
    log.info("spiral model generated!", params = model)

    optimizer = nnx.Optimizer(
        model, optax.adam(settings.training.learning_rate), wrt=nnx.Param
    )
    
    spiral_train(model, optimizer, data, settings.training, np_rng)
    log.info("finished training")

    spiral_plot(model, data, settings.plotting)
    log.info("finished plotting")

    
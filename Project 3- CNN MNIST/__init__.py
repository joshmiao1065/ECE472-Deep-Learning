import jax
import numpy as np
import structlog
import optax

from flax import nnx

from .config import load_settings
from .data import MNIST
from .model import Classify
from .logging import configure_logging
from .training import MNIST_train
from .test import test_accuracy

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

    data = MNIST(
        rng = np_rng,
        train_ratio = settings.data.train_ratio,
    )
    log.info("MNIST data retrieved", shape = data.x_train.shape)

    model = Classify(
        rngs = nnx.Rngs(params=model_key),
        num_classes = settings.model.num_classes,
        shape = settings.model.shape,
        dropout_rate = settings.model.dropout_rate,
        input_depth = settings.model.input_depth,
        layer_depths = settings.model.layer_depth,
        layer_kernel_sizes = settings.model.layer_kernel_sizes,
        strides = settings.model.strides,
        L2_weight = settings.training.L2_weight,
    )

    log.info("spiral model generated!", params = model)

    optimizer_schedule = optax.cosine_decay_schedule(settings.training.learning_rate, settings.training.num_iters,) #same schedule as you
    optimizer = nnx.Optimizer(
        model, optax.adam(optimizer_schedule), wrt=nnx.Param
    )
    
    MNIST_train(model, optimizer, data, settings.training, np_rng)
    log.info("training concluded")

    validation_accuracy = test_accuracy(
        data = data,
        batch_size = settings.training.batch_size,
        model = model,
        validation = True,
    )

    if validation_accuracy >= 0.96:
        log.info("passed validation w/ acc > 0.96")
        real_test_accuracy = test_accuracy(
            data = data,
            batch_size = settings.training.batch_size,
            model = model,
            validation = False,
        )
        log.info("real test accuracy:", accuracy = real_test_accuracy)
        if real_test_accuracy >= 0.955:
            log.info("yippeeeee")
        else:
            log.info("you failed the test bruh")
    else:
        log.info("failed validation noooooooooooooooooooooo")
    
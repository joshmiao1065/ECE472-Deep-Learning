import jax
import numpy as np
import structlog
import optax
from pathlib import Path
import orbax.checkpoint as ocp
from flax import nnx

from .config import load_settings
from .data import cifar10
from .model import Classify
from .logging import configure_logging
from .training import cifar10_train
from .test import test_accuracy, test_top5

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

    data = cifar10(
        rng = np_rng,
        train_ratio = settings.data.train_ratio,
    )
    if settings.model.num_classes == 10:
        log.info("cifar10 data retrieved", shape = data.x_train.shape)
    else:
        log.info("cifar100 data retrieved", shape = data.x_train.shape)

    model = Classify(
        rngs = nnx.Rngs(params=model_key),
        num_classes = settings.model.num_classes,
        shape = settings.model.shape,
        input_depth = settings.model.input_depth,
        layer_depths = settings.model.layer_depth,
        layer_kernel_sizes = settings.model.layer_kernel_sizes,
        strides = settings.model.strides,
        L2_weight = settings.training.L2_weight,
    )
    
    optimizer_schedule = optax.cosine_decay_schedule(settings.training.learning_rate, settings.training.num_iters,) #same schedule as you
    optimizer = nnx.Optimizer(
        model, optax.adam(optimizer_schedule), wrt=nnx.Param
    )

    cifar10_train(model, optimizer, data, settings.training, np_rng)
    log.info("training concluded")

    ckpt_dir = ocp.test_utils.erase_and_create_empty("/saves")
    graphdef, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(ckpt_dir/ "cifar10", state)
    log.info("saved most recent checkpoint")

    #validation and test set and stuff
    if settings.model.num_classes == 10:
        validation_accuracy = test_accuracy(
            data = data,
            batch_size = settings.training.batch_size,
            model = model,
            validation = True,
        )
        log.info("validation accuracy:", val_accuracy = validation_accuracy)
        real_test_accuracy = test_accuracy(
            data = data,
            batch_size = settings.training.batch_size,
            model = model,
            validation = False,
        )
        log.info("real test accuracy:", accuracy = real_test_accuracy)

    else:
        validation_accuracy = test_top5(
            data = data,
            batch_size = settings.training.batch_size,
            model = model,
            validation = True,
        )
        log.info("validation top5 accuracy:", val_accuracy = validation_accuracy)
        real_test_accuracy = test_top5(
            data = data,
            batch_size = settings.training.batch_size,
            model = model,
            validation = False,
        )
        log.info("real test top5 accuracy:", accuracy = real_test_accuracy)

    if real_test_accuracy >= 0.85:
        log.info("yippeeeee, close enough to state of the art i give up")
    else:
        log.info("this is not state of the art :(")

def test10():
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(472) #472 for ECE 472
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.array(data_key))

    log.info("evalutating on test set for CIFAR10!!")

    model = Classify(
        rngs = nnx.Rngs(params=model_key),
        num_classes = settings.model.num_classes,
        shape = settings.model.shape,
        input_depth = settings.model.input_depth,
        layer_depths = settings.model.layer_depth,
        layer_kernel_sizes = settings.model.layer_kernel_sizes,
        strides = settings.model.strides,
        L2_weight = settings.training.L2_weight,
    )

    data = cifar10(
        rng = np_rng,
        train_ratio = settings.data.train_ratio,
    )

    ckpt_dir = Path("/saves").resolve()
    checkpointer = ocp.StandardCheckpointer()
    graphdef, state = nnx.split(model)
    state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
    restored = checkpointer.restore(ckpt_dir/ 'most_recent', state) 
    model = nnx.merge(graphdef, restored)
    log.info("restored model from checkpoint", model = model)

    real_test_accuracy = test_accuracy(
            data = data,
            batch_size = settings.training.batch_size,
            model = model,
            validation = False,
        )
    log.info("real test accuracy:", accuracy = real_test_accuracy)

def test100():
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(472) #472 for ECE 472
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.array(data_key))

    log.info("evalutating on test set for CIFAR100!!")

    model = Classify(
        rngs = nnx.Rngs(params=model_key),
        num_classes = settings.model.num_classes,
        shape = settings.model.shape,
        input_depth = settings.model.input_depth,
        layer_depths = settings.model.layer_depth,
        layer_kernel_sizes = settings.model.layer_kernel_sizes,
        strides = settings.model.strides,
        L2_weight = settings.training.L2_weight,
    )

    data = cifar10( #should be cifar100 but im too lazy to change
        rng = np_rng,
        train_ratio = settings.data.train_ratio,
    )
    ckpt_dir = Path("/saves").resolve()
    checkpointer = ocp.StandardCheckpointer()
    graphdef, state = nnx.split(model)
    state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
    restored = checkpointer.restore(ckpt_dir/ 'most_recent', state) 
    model = nnx.merge(graphdef, restored)
    log.info("restored model from checkpoint", model = model)

    real_test_accuracy = test_accuracy(
            data = data,
            batch_size = settings.training.batch_size,
            model = model,
            validation = False,
        )
    log.info("real test accuracy:", accuracy = real_test_accuracy)
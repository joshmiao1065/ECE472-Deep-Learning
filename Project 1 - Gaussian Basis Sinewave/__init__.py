import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .config import load_settings
from .data import Data
from .logging import configure_logging
from .model import GaussianRegressor
from .plotting import plot_fit, plot_basis_functions
from .training import train

log = structlog.get_logger()


def main() -> None:
    """CLI entry point for hw01 Gaussian-basis regression."""
    settings = load_settings()
    configure_logging()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(472)  # seed is 472 for ECE472 hehe
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(int(np.array(data_key)[0]))

    log.info(
        "Creating dataset",
        num_samples=settings.data.num_samples,
        sigma=settings.data.sigma_noise,
    )
    data = Data(
        rng=np_rng,
        num_samples=settings.data.num_samples,
        sigma=settings.data.sigma_noise,
    )

    # Build model
    log.info(
        "Creating model",
        num_basis=settings.model.num_basis,
        init_sigma=settings.model.init_basis_sigma,
    )
    model = GaussianRegressor(
        rngs=nnx.Rngs(params=model_key),
        num_basis=settings.model.num_basis,
        init_sigma=settings.model.init_basis_sigma,
    )
    log.debug("Initial model params:", model_params=model.model)

    # time to define how to optimize
    optimizer = nnx.Optimizer(
        model, optax.sgd(settings.training.learning_rate), wrt=nnx.Param
    )
    log.info("Optimizer created", optimizer="sgd", lr=settings.training.learning_rate)

    # time to train
    train(model, optimizer, data, settings.training, np_rng)
    log.debug("Trained model params", model_params=model.model)

    # time to plot
    log.info("Generating plots")
    plot_fit(model, data, settings.plotting)
    plot_basis_functions(model, data, settings.plotting)
    log.info("plots generated in folder titled outputs")
    log.info("hw01 run complete!!!!!!")


if __name__ == "__main__":
    main()

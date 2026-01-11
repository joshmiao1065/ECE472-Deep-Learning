import jax
import numpy as np
import structlog
import optax

from flax import nnx
from numpy.random import Generator, PCG64

from .config import load_settings
from .data import Data
from .model import MLP, SparseAutoEncoder
from .logging import configure_logging
from .plotting import spiral_plot, latent_features_plot
from .training import spiral_train, sae_train

"""
Discussion:
The Sparse Autoencoder has been trained to learn features about the Spirals MLP. The SAE was trained on a dimension of 2048.
The features learned by the SAE seem to strongly correspond with particular spirals, as shown in the graphs.
Features 1788 and 1047 strongly correlated with the blue spiral while features 1688and 1091 strongly correlated with the red spiral. 
The other activated features corresponded most strongly to the corners of the graph. I feel like this could be explained because the 
corners are the regions that are easiest to identify as belonging to a certain spiral and thus be tied to blue for the bottom corners 
and red to the top corners.its worth nothing that the magnitudes of the activation are signficantly stronger for the first 5 features 
which highlight an entire decision boundary. it seems like everything in the spirals are just mapped to a handful of features and the 
rest of the features map to either the corners or very minutes parts of the spirals.
"""

def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(472) #472 for ECE 472
    data_key, model_key, sae_key = jax.random.split(key, 3)
    np_rng = np.random.default_rng(np.array(data_key))

    #Generate spiral data
    data = Data(
        rng = Generator(PCG64(seed=472)),
        num_points = settings.data.num_samples // 2,
        num_turns = 4,
        scale = 1.0,
        noise = settings.data.noise,
    )
    log.info("spirals generated", shape = data.x.shape)

    # Build and train MLP classifier
    model = MLP(
        rngs=nnx.Rngs(params=model_key), 
        num_inputs = settings.data.num_features,
        num_outputs = settings.data.num_outputs,
        num_hl=settings.model.num_hl,
        hl_width=settings.model.hl_width,
    )
    log.info("spiral model generated!", 
             num_hl=settings.model.num_hl, 
             hl_width=settings.model.hl_width)

    # MLP optimizer with optional cosine decay
    if settings.mlp_training.use_cosine_decay:
        mlp_schedule = optax.cosine_decay_schedule(
            init_value=settings.mlp_training.learning_rate,
            decay_steps=settings.mlp_training.num_iters,
            alpha=settings.mlp_training.lr_decay_alpha
        )
        mlp_optimizer = nnx.Optimizer(model, optax.adam(mlp_schedule), wrt=nnx.Param)
    else:
        mlp_optimizer = nnx.Optimizer(
            model, optax.adam(settings.mlp_training.learning_rate), wrt=nnx.Param
        )
    
    spiral_train(model, mlp_optimizer, data, settings.mlp_training, np_rng)
    log.info("finished training MLP")

    spiral_plot(model, data, settings.plotting)
    log.info("finished plotting decision boundary")

    # Build and train Sparse Autoencoder
    sae = SparseAutoEncoder(
        rngs=nnx.Rngs(params=sae_key),
        hidden_layer_width=settings.model.hl_width,
        latent_dim=settings.model.latent_dim,
    )
    log.info("SAE model generated!", 
             hidden_width=settings.model.hl_width, 
             latent_dim=settings.model.latent_dim,
             expansion_factor=settings.model.latent_dim / settings.model.hl_width)

    # SAE optimizer with optional cosine decay
    if settings.sae_training.use_cosine_decay:
        sae_schedule = optax.cosine_decay_schedule(
            init_value=settings.sae_training.learning_rate,
            decay_steps=settings.sae_training.num_iters,
            alpha=settings.sae_training.lr_decay_alpha
        )
        sae_optimizer = nnx.Optimizer(sae, optax.adam(sae_schedule), wrt=nnx.Param)
    else:
        sae_optimizer = nnx.Optimizer(
            sae, optax.adam(settings.sae_training.learning_rate), wrt=nnx.Param
        )

    sae_train(model, sae, sae_optimizer, data, settings.sae_training, np_rng)
    log.info("finished training SAE")

    latent_features_plot(model, sae, data, settings.plotting)
    log.info("finished plotting latent features")
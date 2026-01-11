import structlog
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax as nnx

from .logging import configure_logging
from .config import load_settings
from .data import prepare_datasets
from .model import TextMLPModel
from .training import create_train_state, train_step, eval_step, train_and_evaluate

log = structlog.get_logger() 

def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed)
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.array(data_key))
    # Prepare data and tokenizer (now returns train, val, test)
    train_ds, val_ds, test_ds, token_iterator, tokenizer = prepare_datasets(settings)
    num_classes = 4

    model = TextMLPModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=settings.model.embed_dim,
        latent_dim=settings.model.latent_dim,
        num_classes=num_classes,
        hidden_dims=settings.model.hidden_dims,
    )
    log.info("model instantiated", vocab_size=tokenizer.vocab_size)

    #i moved tthe optimizer that would be here to training.py so __init__ bc i wanna try using state to hold everything
    # Create training state (parameters + optimizer)
    rng = jax.random.PRNGKey(settings.random_seed)
    state = create_train_state(rng, model, settings.training.learning_rate, getattr(settings.training, 'weight_decay', 0.0))

    # Prepare RNG for dropout and other stochastic components
    rng, dropout_rng = jax.random.split(rng)

    # Run the high-level training + evaluation loop
    state = train_and_evaluate(state, train_ds, val_ds, test_ds, token_iterator, settings, dropout_rng)
    print("Training complete")
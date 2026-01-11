import structlog

import jax
import jax.numpy as jnp
import numpy as np

import optax

from tqdm import trange
from flax import nnx

from flax.training import train_state

log = structlog.get_logger() 

class TrainState(train_state.TrainState):
    """
    mini checkpointing helper to hold my model parameters, optimizer transformer/state
    """
    pass

def create_train_state(rng, model, learning_rate, weight_decay):
    # Initialize model parameters with dummy input and mask to infer input shapes
    dummy_input = jnp.ones((1, 128), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, 128), dtype=jnp.int32)
    params = model.init(rng, dummy_input, dummy_mask)
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state

@jax.jit
def train_step(state, batch, dropout_rng):

    def loss_fn(params):
        #cross entropy loss function
        logits, z = state.apply_fn(params, batch["input_ids"], batch.get("attention_mask", None), deterministic=False, rngs={"dropout": dropout_rng})
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, batch["labels"]).mean()
        return ce, (logits, z)

    (loss, (logits, z)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    acc = jnp.mean(jnp.argmax(logits, -1) == batch["labels"])
    return state, loss, acc, logits, z

def train_loop(state, train_iter, num_steps=None, print_every=100):

    bar = trange(num_steps or 0) if num_steps is not None else None
    step = 0
    for batch in train_iter:
        batch_jax = {k: jnp.array(v) for k, v in batch.items()}
        state, loss, acc, _, _ = train_step(state, batch_jax, jax.random.PRNGKey(472))
        if step % print_every == 0:
            print(f"Step {step} | Loss: {float(loss):.4f} | Acc: {float(acc):.4f}")
        step += 1
        if num_steps is not None and step >= num_steps:
            break
        if bar is not None:
            bar.update(1)
    return state


def train_and_evaluate(state, train_ds, val_ds, test_ds, token_iterator, settings, rng, patience: int = 3):
    """
    5 epochs over training split and evalues on validation split after each epoch.
    I decided to be ambition and utilize early stopping if val accuracy doesn't improve for 3 epochs
    """
    step = 0
    best_params = state.params
    best_val = -1.0
    wait = 0
    for epoch in range(settings.training.num_epochs):
        print(f"Epoch {epoch+1}/{settings.training.num_epochs}")
        train_iter = token_iterator(train_ds, settings.training.batch_size, shuffle=True)
        # Use tqdm for per-epoch progress
        n_batches = (len(train_ds) + settings.training.batch_size - 1) // settings.training.batch_size
        with trange(n_batches, desc=f"Epoch {epoch+1}") as pbar:
            for i, batch in enumerate(train_iter):
                batch_jax = {k: jnp.array(v) for k, v in batch.items()}
                # split RNG for this training step and use it for dropout
                rng, step_rng = jax.random.split(rng)
                state, loss, acc, _, _ = train_step(state, batch_jax, step_rng)
                if step % 100 == 0:
                    pbar.set_postfix({'loss': float(loss), 'acc': float(acc)})
                step += 1
                pbar.update(1)
        # Eval on validation set at epoch end
        val_iter = token_iterator(val_ds, settings.training.batch_size, shuffle=False)
        losses = []
        accs = []
        for batch in val_iter:
            batch_jax = {k: jnp.array(v) for k, v in batch.items()}
            loss, acc, _, _ = eval_step(state, batch_jax)
            losses.append(float(loss))
            accs.append(float(acc))
        val_loss = np.mean(losses)
        val_acc = np.mean(accs)
        print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

        # Early stopping on validation accuracy
        if val_acc > best_val:
            best_val = val_acc
            best_params = state.params
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping after {epoch+1} epochs (no improvement for {patience} epochs)")
                break

    # Restore best params
    state = state.replace(params=best_params)

    # Final evaluation on test set
    test_iter = token_iterator(test_ds, settings.training.batch_size, shuffle=False)
    losses = []
    accs = []
    for batch in test_iter:
        batch_jax = {k: jnp.array(v) for k, v in batch.items()}
        loss, acc, _, _ = eval_step(state, batch_jax)
        losses.append(float(loss))
        accs.append(float(acc))
    print(f"Test loss: {np.mean(losses):.4f}, Test acc: {np.mean(accs):.4f}")

    return state

@jax.jit
def eval_step(state, batch):
    """
    Run a single evaluation step. dropout disabled here
    """
    logits, z = state.apply_fn(state.params, batch["input_ids"], batch.get("attention_mask", None), deterministic=True)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["labels"]).mean()
    acc = jnp.mean(jnp.argmax(logits, -1) == batch["labels"])
    return loss, acc, logits, z
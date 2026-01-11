import jax.numpy as jnp
import numpy as np
import structlog
import optax

from .model import Classify
from .data import cifar10

log = structlog.get_logger()

def test_accuracy(
    model: Classify,
    data: cifar10,
    batch_size: int,
    validation: bool = True
) -> float:
    
    if validation == True:
        x_np, y_np = data.get_validation()
    else:
        x_np, y_np = data.get_test()
        
    num_samples = x_np.shape[0]

    # ensure labels are 1-D integer arrays
    y_np = np.asarray(y_np).reshape(-1)
    
    #batch predictions
    preds = []
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        xb = jnp.asarray(x_np[start:end])
        logits = model(xb, train=False)
        batch_pred = jnp.argmax(logits, axis=-1)
        preds.append(batch_pred)
    
    #compute accuracy
    all_preds = jnp.concatenate(preds, axis=0)
    all_labels = jnp.asarray(y_np)
    correct = jnp.sum(all_preds == all_labels)
    accuracy = correct / num_samples
    
    log.info(
        "Evaluated %s set: %d samples, %d correct, Accuracy=%.2f%%",
        "validation" if validation else "test",
        int(num_samples),
        int(correct),
        float(accuracy * 100),
    )
    
    return float(accuracy)

def test_top5(
    model: Classify,
    data: cifar10,
    batch_size: int,
    validation: bool = True
) -> float:
    """Compute top-5 accuracy on the validation or test set."""
    
    if validation:
        x_np, y_np = data.get_validation()
    else:
        x_np, y_np = data.get_test()
        
    num_samples = x_np.shape[0]

    # ensure labels are 1-D integer arrays
    y_np = np.asarray(y_np).reshape(-1)
    
    correct_top5 = 0

    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        xb = jnp.asarray(x_np[start:end])
        yb = jnp.asarray(y_np[start:end])

        logits = model(xb, train=False)

        # get indices of top 5 predictions per sample
        top5 = jnp.argsort(logits, axis=-1)[:, -5:]  # shape (batch_size, 5)

        # check if true label is in top 5 predictions
        correct_batch = jnp.any(top5 == yb[:, None], axis=-1)
        correct_top5 += jnp.sum(correct_batch)

    top5_accuracy = correct_top5 / num_samples

    log.info(
        "Evaluated %s set: %d samples, %d correct (top-5), Accuracy=%.2f%%",
        "validation" if validation else "test",
        int(num_samples),
        int(correct_top5),
        float(top5_accuracy * 100),
    )

    return float(top5_accuracy)
import jax.numpy as jnp
import numpy as np
import structlog
import optax

from .model import Classify
from .data import MNIST

log = structlog.get_logger()

def test_accuracy(
    model: Classify,
    data: MNIST,
    batch_size: int,
    validation: bool = True
) -> float:
    
    if validation == True:
        x_np, y_np = data.get_validation()
    else:
        x_np, y_np = data.get_test()
        
    num_samples = x_np.shape[0]
    
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
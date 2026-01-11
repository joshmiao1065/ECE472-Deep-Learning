import jax.numpy as jnp
import numpy as np
import structlog

from flax import nnx
from tqdm import trange
from .config import MLPTrainingSettings, SAETrainingSettings
from .data import Data
from .model import MLP, SparseAutoEncoder

log = structlog.get_logger()

@nnx.jit
def train_step_spiral(
    model: MLP,
    optimizer: nnx.Optimizer,
    x: jnp.ndarray,
    target: jnp.ndarray,
    epsilon: float,
 ):
    """Single training step for MLP classifier."""
    def loss_function(model: MLP):
        probs = model(x)
        #limit the max and min value of value passed into logarithm to avoid nan loss
        probs = jnp.where(probs < epsilon, epsilon, probs)
        probs = jnp.where(probs > 1 - epsilon, 1 - epsilon, probs)

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
    settings: MLPTrainingSettings,
    np_rng: np.random.Generator,
):
    """SGD training for MLP classifier."""
    log.info("Start training MLP", 
             num_iters=settings.num_iters, 
             batch_size=settings.batch_size,
             learning_rate=settings.learning_rate)
    bar = trange(settings.num_iters, desc="Training MLP")
    for i in bar:
        x_np, target_np = data.get_batch(np_rng, settings.batch_size)
        x, target = jnp.asarray(x_np), jnp.asarray(target_np)
        target = target.reshape(-1, 1)  #reshape target to correct output dimensions

        loss = train_step_spiral(model, optimizer, x, target, settings.epsilon)
        
        bar.set_description(f"MLP Loss @ {i} => {loss:.6f}")
        
        if i % 1000 == 0:
            log.debug("MLP training progress", iteration=i, loss=float(loss))
        
        bar.refresh()
    log.info("MLP Training Finished", final_loss=float(loss))


@nnx.jit
def train_step_sae(
    sae: SparseAutoEncoder,
    optimizer: nnx.Optimizer,
    z: jnp.ndarray,
    lambda_sparsity: float,
):
    """Single training step for Sparse Autoencoder."""
    def loss_function(sae: SparseAutoEncoder):
        z_hat, h = sae(z)
        
        # Reconstruction loss: MSE between original and reconstructed hidden states
        reconstruction_loss = jnp.mean((z - z_hat) ** 2)
        
        # Sparsity loss: L1 penalty on latent activations
        sparsity_loss = jnp.mean(jnp.abs(h))
        
        # Combined loss
        total_loss = reconstruction_loss + lambda_sparsity * sparsity_loss
        
        return total_loss, (reconstruction_loss, sparsity_loss)
    
    (total_loss, (recon_loss, sparse_loss)), grads = nnx.value_and_grad(
        loss_function, has_aux=True
    )(sae)
    
    optimizer.update(sae, grads)
    return total_loss, recon_loss, sparse_loss


def sae_train(
    mlp: MLP,
    sae: SparseAutoEncoder,
    optimizer: nnx.Optimizer,
    data: Data,
    settings: SAETrainingSettings,
    np_rng: np.random.Generator,
):
    """Train Sparse Autoencoder on frozen MLP hidden states."""
    log.info("Start training SAE on frozen MLP", 
             num_iters=settings.num_iters, 
             batch_size=settings.batch_size,
             learning_rate=settings.learning_rate,
             lambda_sparsity=settings.lambda_sparsity)
    
    # Extract all hidden states from frozen MLP
    all_x = jnp.asarray(data.x)
    log.debug("Extracting hidden states from MLP", num_samples=all_x.shape[0])
    all_z = mlp.extract_final_hidden_state(all_x)
    log.debug("Hidden states extracted", shape=all_z.shape)
    
    bar = trange(settings.num_iters, desc="Training SAE")
    for i in bar:
        # Sample random batch of hidden states
        choices = np_rng.choice(all_z.shape[0], size=settings.batch_size, replace=False)
        z_batch = all_z[choices]
        
        total_loss, recon_loss, sparse_loss = train_step_sae(
            sae, optimizer, z_batch, settings.lambda_sparsity
        )
        
        bar.set_description(
            f"SAE Loss @ {i} => L_tot={total_loss:.4f} | L_recon={recon_loss:.4f} | L_sparse={sparse_loss:.4f}"
        )
        
        if i % 1000 == 0:
            log.debug("SAE training progress", 
                     iteration=i, 
                     total_loss=float(total_loss),
                     recon_loss=float(recon_loss),
                     sparse_loss=float(sparse_loss))
        
        bar.refresh()
    
    log.info("SAE Training Finished", 
             final_total_loss=float(total_loss),
             final_recon_loss=float(recon_loss),
             final_sparse_loss=float(sparse_loss))
import jax
import jax.numpy as jnp
from flax import nnx
import structlog

log = structlog.get_logger()

def xavier_kernel_init(num_inputs: int, num_outputs: int):
    """Glorot/Xavier uniform initialization"""
    def init(rng, shape, dtype=jnp.float32):
        limit = jnp.sqrt(6.0 / (num_inputs + num_outputs))
        return jax.random.uniform(rng, shape, dtype, minval=-limit, maxval=limit)
    return init

class Block(nnx.Module):
    """define each hidden layer block"""
    def __init__(self, num_inputs: int, hl_width: int, rngs: nnx.Rngs):
        super().__init__()
        self.linear = nnx.Linear(
            num_inputs,
            hl_width,
            rngs=rngs,
            kernel_init=xavier_kernel_init(num_inputs, hl_width),
        )

    def __call__(self, x: jax.Array):
        x = self.linear(x)
        return jax.nn.relu(x)

class MLP(nnx.Module):
    """build our hidden layers"""
    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        num_inputs: int,
        num_outputs: int,
        num_hl: int,
        hl_width: int,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hl = num_hl
        self.hl_width = hl_width

        key = rngs.params()

        #initial layer
        k_first, key = jax.random.split(key)
        self.first = nnx.Linear(
            num_inputs,
            hl_width,
            rngs=nnx.Rngs(k_first),
            kernel_init=xavier_kernel_init(num_inputs, hl_width),
        )

        #hidden layers
        @nnx.split_rngs(splits=num_hl - 1)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def make_block(rngs_for_one_layer: nnx.Rngs):
            return Block(num_inputs=hl_width, hl_width=hl_width, rngs=rngs_for_one_layer)

        self.hidden_blocks = make_block(nnx.Rngs(key))

        #output layer
        k_out = jax.random.split(key, 2)[1]
        self.out = nnx.Linear(
            hl_width,
            num_outputs,
            rngs=nnx.Rngs(k_out),
            kernel_init=xavier_kernel_init(hl_width, num_outputs),
        )

    def extract_final_hidden_state(self, x: jax.Array) -> jax.Array:
        """Extract activations before the output layer for SAE training."""
        # First transform
        x = jax.nn.relu(self.first(x))

        #applies Block sequentially to our layers
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def forward(carry, block):
            return block(carry)

        x = forward(x, self.hidden_blocks) 
        return x

    def __call__(self, x: jax.Array):
        # Extract final hidden state
        x = self.extract_final_hidden_state(x)
        
        #use logits as an input to the sigmoid function in our last matmul
        logits = self.out(x)  
        probs = jax.nn.sigmoid(logits)
        return probs


class SparseAutoEncoder(nnx.Module):
    """Sparse Autoencoder for discovering interpretable features."""
    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        hidden_layer_width: int,
        latent_dim: int,
    ):
        super().__init__()
        self.hidden_layer_width = hidden_layer_width
        self.latent_dim = latent_dim
        
        key = rngs.params()
        k_encoder, k_decoder = jax.random.split(key, 2)
        
        # Encoder: compress hidden states to sparse latent space
        self.encoder = nnx.Linear(
            hidden_layer_width,
            latent_dim,
            rngs=nnx.Rngs(k_encoder),
            kernel_init=xavier_kernel_init(hidden_layer_width, latent_dim),
        )
        
        # Decoder: reconstruct hidden states from latent space
        self.decoder = nnx.Linear(
            latent_dim,
            hidden_layer_width,
            rngs=nnx.Rngs(k_decoder),
            kernel_init=xavier_kernel_init(latent_dim, hidden_layer_width),
        )

    def encode(self, z: jax.Array) -> jax.Array:
        """Encode hidden states to latent space with ReLU activation."""
        return jax.nn.relu(self.encoder(z))

    def decode(self, h: jax.Array) -> jax.Array:
        """Decode latent activations back to hidden state space."""
        return self.decoder(h)

    def __call__(self, z: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Forward pass: returns (reconstruction, latent_activations)."""
        h = self.encode(z)
        z_hat = self.decode(h)
        return z_hat, h
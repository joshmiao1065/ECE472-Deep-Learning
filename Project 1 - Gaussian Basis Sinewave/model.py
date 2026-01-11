import jax
import jax.numpy as jnp
import numpy as np
import structlog
from flax import nnx
from dataclasses import dataclass
from typing import Dict

log = structlog.get_logger()


@dataclass
class Parameters:
    """Container for logging / inspection of learned parameters."""

    mus: np.ndarray
    sigmas: np.ndarray
    weights: np.ndarray
    bias: float


class Linear(nnx.Module):
    """
    maps M basis outputs to scalaras
    """

    def __init__(self, *, rngs: nnx.Rngs, num_basis: int):
        log.info("Initializing Linear module", num_basis=num_basis)
        key = rngs.params()
        # small initialization scale
        self.w = nnx.Param(0.1 * jax.random.normal(key, (num_basis,)))
        """
        for the longest time i thought jnp.zeros() was sufficient to create a scalar. 
        i didnt know i had to do jnp.zeros(()) 😭😭😭😭
        and yes i copy pasted this emoji just to emphasize my struggle
        """
        self.b = nnx.Param(jnp.zeros(()))
        log.debug(
            "Linear params initialized",
            w_shape=self.w.value.shape,
            b_shape=self.b.value.shape,
        )

    def __call__(self, phi: jax.Array) -> jax.Array:
        """
        phi shape: (batch, M) -> returns (batch,)
        """
        return jnp.dot(phi, self.w.value) + self.b.value

    # learned what decorators and warppers are today are you proud professor?
    @property
    def model(self) -> Dict[str, jnp.ndarray]:
        return {"w": self.w.value, "b": self.b.value}


class BasisExpansion(nnx.Module):
    """
    Gaussian basis functions with learnable centers mus and sigmas.
    phi_j(x | mu_j, sigma_j) = exp(-(x - mu_j)^2 / sigma_j^2)
    """

    def __init__(self, *, rngs: nnx.Rngs, num_basis: int, init_sigma: float = 0.1):
        log.info(
            "Initializing BasisExpansion", num_basis=num_basis, init_sigma=init_sigma
        )
        key = rngs.params()
        # split to get reproducible mu and sigma inits
        k1, k2 = jax.random.split(key, 2)

        # initialize mu evenly across domain [0,1] with small noise
        mu_init = jnp.linspace(0.0, 1.0, num_basis)
        mu_init = mu_init + 0.01 * jax.random.normal(
            k1, (num_basis,)
        )  # small perturbation  to vary basis functions initially
        self.mu = nnx.Param(mu_init)

        # lets convert sigma to log in case it gets too small or negative
        log_sigma_init = jnp.log(jnp.ones(num_basis) * 0.05)  # smaller spread
        self.log_sigma = nnx.Param(log_sigma_init)

        # idk how this debug statement got this long. it was revolutionary when i realized that i could put if statements in here
        log.debug(
            "Basis params initialized",
            mu_init=mu_init.tolist()
            if isinstance(mu_init, jnp.ndarray)
            else str(mu_init),
            log_sigma_init=log_sigma_init.tolist()
            if isinstance(log_sigma_init, jnp.ndarray)
            else str(log_sigma_init),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Compute phi for inputs x.

        x: shape (batch, 1) or (batch,)
        returns phi: (batch, M)
        """
        x = jnp.squeeze(x)  # -> (batch,)
        log.debug("x squeezed:", shape=x.shape)
        x = x[:, None]  # -> (batch, 1)
        log.info("x shape:", shape=x.shape)
        mu = self.mu.value[None, :]  # -> (1, M)
        log.info("mu shape:", shape=mu.shape)
        # i think this is the right place to exponentiate and get rid of the log for sigma
        sigma = jnp.exp(self.log_sigma.value)
        sigma = sigma[None, :]
        log.info("sigma shape:", shape=sigma.shape)

        diff = x - mu  # (batch, M) i think
        phi = jnp.exp(-(diff**2) / (sigma**2))
        return phi

    @property
    def model(self) -> Dict[str, jnp.ndarray]:
        return {"mu": self.mu.value, "sigma": jnp.exp(self.log_sigma.value)}


class GaussianRegressor(nnx.Module):
    """
    build the regression model:
       y_hat(x) = sum_j w_j * phi_j(x | mu_j, sigma_j) + b
    """

    def __init__(self, *, rngs: nnx.Rngs, num_basis: int, init_sigma: float = 0.1):
        log.info(
            "Initializing GaussianRegressor", num_basis=num_basis, init_sigma=init_sigma
        )
        key = rngs.params()
        # split RNG for submodules
        k_basis, k_linear = jax.random.split(key, 2)
        self.basis = BasisExpansion(
            rngs=nnx.Rngs(params=k_basis), num_basis=num_basis, init_sigma=init_sigma
        )
        self.linear = Linear(rngs=nnx.Rngs(params=k_linear), num_basis=num_basis)
        log.info("GaussianRegressor initialized")

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        x: (batch, 1) -> returns (batch,)
        """
        phi = self.basis(x)
        return self.linear(phi)

    @property
    def model(self) -> Parameters:
        mus = np.array(self.basis.mu.value)
        sigmas = np.array(jnp.exp(self.basis.log_sigma.value))
        weights = np.array(self.linear.w.value)
        bias = float(np.array(self.linear.b.value).squeeze())
        return Parameters(mus=mus, sigmas=sigmas, weights=weights, bias=bias)

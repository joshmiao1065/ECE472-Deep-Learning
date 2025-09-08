import jax.numpy as jnp
import jax.random as random


def init_basis_params(key, M):
    # initialize our M Gaussian basis funcs:
    # mu_j ~ U(0,1)
    # signma_j = .1

    mu = random.uniform(key, (M,), minval=0.0, maxval=1.0)
    sigma = (
        jnp.ones((M,)) * 0.1
    )  # took me embarassingly long to create a vector of .1s lol
    return {"mu": mu, "sigma": sigma}  # we <3 dictionaries


def basis_expansion(params, x):
    # phi_j(x) = exp(-((x-mu_j)^2) / sigma_j^2), j=1 to M

    mu = params["mu"]
    sigma = params["sigma"]
    x_expanded = x[..., None]  # ensure x is broadcastable against mu (M,)
    diff = x_expanded - mu
    phi = jnp.exp(-(diff**2) / (sigma**2))
    return phi

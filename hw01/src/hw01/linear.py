import jax.numpy as jnp
import jax.random as random


def init_linear_params(key, M):
    # initialize weights and bias type shit

    w_key, _ = random.split(key)
    w = (
        random.normal(w_key, (M,)) * 0.1
    )  # i think we need to start w small weights hence .1 scale factor
    b = jnp.zeros(())
    return {"w": w, "b": b}  # we love dictionaries


def linear_forward(params, phi):
    # y= w*phi + b
    return jnp.dot(phi, params["w"]) + params["b"]

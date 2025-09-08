import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import structlog

from .linear import init_linear_params, linear_forward
from .basis_expansion import init_basis_params, basis_expansion

log = structlog.get_logger("hw01")


def main():
    # data generation(sin wave)
    N = 50
    sigma_noise = 0.1
    key = jax.random.PRNGKey(472)  # seed is 472 for ECE472
    key, x_key, eps_key = jax.random.split(key, 3)
    x = jax.random.uniform(x_key, (N,), minval=0.0, maxval=1.0)
    y = jnp.sin(2 * jnp.pi * x) + sigma_noise * jax.random.normal(eps_key, (N,))

    xs = jnp.linspace(0.0, 1.0, 200)
    ys_true = jnp.sin(2 * jnp.pi * xs)

    # initialize M basis and linear layer
    M = 16
    key, b_key, l_key = jax.random.split(key, 3)
    basis_params = init_basis_params(b_key, M)
    linear_params = init_linear_params(l_key, M)
    params = {"basis": basis_params, "linear": linear_params}

    # loss function definition and prediction
    def predict(p, x_in):
        phi = basis_expansion(p["basis"], x_in)
        return linear_forward(p["linear"], phi)

    def loss_fn(p, x_batch, y_batch):
        y_pred = predict(p, x_batch)
        return 0.5 * jnp.mean((y_pred - y_batch) ** 2)

    grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))

    # train w stochastic gradient descent🤓 will the emoji print okay?
    lr = 0.1
    epochs = 1000
    batch_size = 10
    num_batches = N // batch_size

    for epoch in range(epochs):
        key, perm_key = jax.random.split(key)
        perm = jax.random.permutation(perm_key, N)
        x_shuf = x[perm]
        y_shuf = y[perm]

        for i in range(num_batches):
            xb = x_shuf[i * batch_size : (i + 1) * batch_size]
            yb = y_shuf[i * batch_size : (i + 1) * batch_size]
            grads = grad_fn(params, xb, yb)

            params["basis"]["mu"] -= lr * grads["basis"]["mu"]
            params["basis"]["sigma"] -= lr * grads["basis"]["sigma"]
            params["linear"]["w"] -= lr * grads["linear"]["w"]
            params["linear"]["b"] -= lr * grads["linear"]["b"]

        if epoch % 100 == 0 or epoch == epochs - 1:
            current_loss = float(loss_fn(params, x, y))
            log.info("Epoch complete", epoch=epoch, loss=current_loss)

    # output graphs
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    y_pred = predict(params, xs)

    # fit vs. true graph
    fig1, ax1 = plt.subplots()
    ax1.scatter(x, y, color="tab:blue", label="Noisy data")
    ax1.plot(xs, ys_true, color="tab:orange", label="True sine")
    ax1.plot(xs, y_pred, color="tab:green", label="Model fit")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    fig1.tight_layout()
    fig1_path = os.path.join(out_dir, "regression_fit.png")
    fig1.savefig(fig1_path)
    log.info("Saved regression fit plot", path=fig1_path)

    # Gaussian basis function graph
    fig2, ax2 = plt.subplots()
    for j in range(M):
        mu_j = float(params["basis"]["mu"][j])
        sigma_j = float(params["basis"]["sigma"][j])
        phi_j = jnp.exp(-((xs - mu_j) ** 2) / (sigma_j**2))
        ax2.plot(xs, phi_j, alpha=0.7)
    ax2.set_xlabel("x")
    ax2.set_ylabel(
        "φ_j(x)"
    )  # i copy and pasted the phi but it just looks so weird in VScode
    ax2.set_title("Gaussian Basis Functions")
    fig2.tight_layout()
    fig2_path = os.path.join(out_dir, "basis_functions.png")
    fig2.savefig(fig2_path)
    log.info("Saved basis functions plot", path=fig2_path)

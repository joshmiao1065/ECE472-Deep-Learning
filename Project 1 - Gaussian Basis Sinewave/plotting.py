import matplotlib
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import structlog
from pathlib import Path

log = structlog.get_logger()
font = {"size": 10}
matplotlib.style.use("classic")
matplotlib.rc("font", **font)


def plot_fit(model: any, data: any, settings: any) -> None:
    """
    Plot the data points, the true noiseless sine, and the model manifold.
    """
    log.info("Plotting fit")
    xs = np.linspace(0.0, 1.0, 500)
    xs_jnp = jnp.asarray(xs)[:, None]
    y_pred = np.array(model(xs_jnp))
    y_true = np.sin(2.0 * np.pi * xs)

    fig, ax = plt.subplots(1, 1, figsize=settings.figsize, dpi=settings.dpi)
    ax.set_title("Noisy data, true sine, and learned manifold")
    ax.set_xlabel("x")
    ax.set_ylabel("y", labelpad=10)
    ax.plot(xs, y_true, "-", label="True (noiseless)")
    ax.plot(xs, y_pred, "-", linewidth=2, label="Model")
    ax.scatter(
        data.x.flatten(), data.y, marker="o", alpha=0.8, label="Noisy observations"
    )
    ax.legend()
    plt.tight_layout()

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(settings.output_dir) / "fit.png"
    plt.savefig(output_path)
    plt.close(fig)
    log.info("Saved fit plot", path=str(output_path))


def plot_basis_functions(model: any, data: any, settings: any) -> None:
    """
    Plot each learned basis function across the input domain.
    """
    log.info("Plotting basis functions")
    xs = np.linspace(0.0, 1.0, 500)
    xs_jnp = jnp.asarray(xs)[:, None]
    # compute raw basis outputs (batch, M)
    phi = np.array(model.basis(xs_jnp))

    fig, ax = plt.subplots(1, 1, figsize=settings.figsize, dpi=settings.dpi)
    M = phi.shape[1]
    for j in range(M):
        ax.plot(xs, phi[:, j], label=f"phi_{j}")
    ax.set_title("Learned Gaussian basis functions")
    ax.set_xlabel("x")
    ax.set_ylabel("phi_j(x)", labelpad=10)
    # i dont think this plot needs a legend so im not gonna make one
    plt.tight_layout()

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(settings.output_dir) / "basis_functions.png"
    plt.savefig(output_path)
    plt.close(fig)
    log.info("Saved basis functions plot", path=str(output_path))

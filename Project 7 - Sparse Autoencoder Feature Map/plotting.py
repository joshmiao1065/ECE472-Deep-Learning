import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import structlog
import jax.numpy as jnp

from sklearn.inspection import DecisionBoundaryDisplay
from tqdm import tqdm
from .config import PlottingSettings
from .data import Data
from .model import MLP, SparseAutoEncoder

log = structlog.get_logger()

font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}
matplotlib.style.use("classic")
matplotlib.rc("font", **font)

def spiral_plot(
    model: MLP,
    data: Data,
    settings: PlottingSettings,
):
    """Plot spiral decision boundary and save to a PDF."""
    log.info("generating spiral_plot")

    x_min, x_max = data.x[:, 0].min() * 1.1, data.x[:, 0].max() * 1.1
    y_min, y_max = data.x[:, 1].min() * 1.1, data.x[:, 1].max() * 1.1

    x_vals = np.linspace(x_min, x_max, settings.linspace)
    y_vals = np.linspace(y_min, y_max, settings.linspace)

    feature1, feature2 = np.meshgrid(x_vals, y_vals)
    
    log.debug("feature1 has dimensions:", shape=feature1.shape)
    log.debug("feature2 has dimensions:", shape=feature2.shape)

    grid = np.vstack([feature1.ravel(), feature2.ravel()]).T
    
    predicted_prob = model(grid)
    predicted_prob = predicted_prob.ravel().reshape(feature1.shape)
    log.debug("predicted_prob has dimensions:", shape=predicted_prob.shape)

    predicted_prob = np.where(predicted_prob > 0.5, 1, 0)
    
    fig, ax = plt.subplots(figsize=settings.figsize, dpi=settings.dpi)
    display = DecisionBoundaryDisplay(xx0 = feature1, xx1 = feature2, response = predicted_prob)
    display.plot(ax=ax, cmap=plt.cm.seismic, alpha=0.5)
    ax.scatter(data.x[:, 0], data.x[:, 1], c=data.target, cmap=plt.cm.coolwarm, edgecolors='k', s=100)

    ax.set_xlabel('x-position')
    ax.set_ylabel('y-position')
    ax.set_title("Decision Boundary of Spirals")

    plt.tight_layout()

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "classification.pdf"
    
    plt.savefig(output_path)
    plt.close()
    
    log.info("Decision boundary plot generated!", path=str(output_path))


def latent_features_plot(
    mlp: MLP,
    sae: SparseAutoEncoder,
    data: Data,
    settings: PlottingSettings,
    num_features_to_plot: int =9,
):
    """Plot activation maps of top sparse latent features."""
    log.info("Plotting Sparse Latent Features")

    x_min, x_max = data.x[:, 0].min() - 1, data.x[:, 0].max() + 1
    y_min, y_max = data.x[:, 1].min() - 1, data.x[:, 1].max() + 1
    
    mesh_step = 0.1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, mesh_step), 
        np.arange(y_min, y_max, mesh_step)
    )
    coords = jnp.column_stack([xx.ravel(), yy.ravel()])

    # Extract hidden states and encode to latent space
    dense_hidden_state_z = mlp.extract_final_hidden_state(coords)
    latent_activations = sae.encode(dense_hidden_state_z)
    
    # Find top features by mean activation
    mean_activations = jnp.mean(latent_activations, axis=0)
    top_indices = jnp.argsort(mean_activations)[::-1][:num_features_to_plot]

    nrows_cols = int(np.ceil(np.sqrt(num_features_to_plot)))
    fig, axes = plt.subplots(
        nrows=nrows_cols, ncols=nrows_cols, figsize=(12, 12), dpi=settings.dpi
    )
    axes = axes.flatten()

    for i, ax in enumerate(tqdm(axes, desc="Generating feature plots")):
        if i >= len(top_indices):
            ax.axis('off')
            continue

        feature_index = top_indices[i]

        feature_activations_1d = latent_activations[:, feature_index]
        feature_map = feature_activations_1d.reshape(xx.shape)

        max_val = jnp.max(feature_activations_1d)
        max_idx = jnp.argmax(feature_activations_1d)
        max_coords = coords[max_idx, :]
        x_at_max = max_coords[0]
        y_at_max = max_coords[1]

        ax.set_title(
            f"N={feature_index} (Max={max_val:.2f} @ ({x_at_max:.1f}, {y_at_max:.1f}))",
            fontsize=10,
        )

        c = ax.pcolormesh(xx, yy, feature_map, cmap='plasma', shading='auto')

        ax.scatter(
            data.x[:, 0],
            data.x[:, 1],
            c=data.target,
            cmap=plt.cm.RdBu,
            edgecolors='k',
            s=5,
            alpha=0.2,
        )

        ax.set_xticks([])
        ax.set_yticks([])

        plt.colorbar(c, ax=ax, orientation='vertical', shrink=0.6)

    plt.suptitle(
        f"Activation Maps of Top {num_features_to_plot} Sparse Latent Features",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "latent_features.pdf"
    plt.savefig(output_path)
    plt.close()
    log.info("Latent features plot generated!", path=str(output_path))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import structlog

from sklearn.inspection import DecisionBoundaryDisplay
from .config import PlottingSettings
from .data import Data
from .model import MLP

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
    log.debug("feature2 has dimensions:", shape=feature1.shape)

    grid = np.vstack([feature1.ravel(), feature2.ravel()]).T
    
    predicted_prob = model(grid)
    predicted_prob = predicted_prob.ravel().reshape(feature1.shape)
    log.debug("predicted_prob has dimensions:", shape=predicted_prob.shape)


    log.debug("target", target=predicted_prob)
    predicted_prob = np.where(predicted_prob > 0.5, 1, 0)
    
    display = DecisionBoundaryDisplay(xx0 = feature1, xx1 = feature2, response = predicted_prob)
    display.plot()
    display.ax_.scatter(data.x[:, 0], data.x[:, 1], c=data.target)

    plt.xlabel('x-position')
    plt.ylabel('y-position')

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "hw02_plt.pdf"
    
    plt.title("Binary Classification of 2 Archimedean Spirals")
    plt.savefig(output_path)
    
    log.info("Plot Generated! FIinally done!!!!!!!", path=str(output_path))
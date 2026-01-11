from dataclasses import InitVar, dataclass, field
import numpy as np
import structlog
from typing import Tuple

log = structlog.get_logger()


@dataclass
class Data:
    """
    Generate noisy sine data:
        x ~ Uniform(0, 1)
        y = sin(2*pi*x) + Normal(0, sigma_noise)
    """

    rng: InitVar[np.random.Generator]
    num_samples: int
    sigma: float

    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    index: np.ndarray = field(init=False)

    def __post_init__(self, rng: np.random.Generator):
        log.info("Generating data", num_samples=self.num_samples, sigma=self.sigma)
        self.index = np.arange(self.num_samples)
        self.x = rng.uniform(0.0, 1.0, size=(self.num_samples, 1))
        clean_y = np.sin(2.0 * np.pi * self.x)
        self.y = rng.normal(loc=clean_y, scale=self.sigma).flatten()
        log.debug(
            "Sample of generated data",
            x_sample=self.x[:5].tolist(),
            y_sample=self.y[:5].tolist(),
        )

    def get_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select a random batch (with replacement if batch_size > num_samples).
        Returns (x_batch, y_batch) where x_batch shape = (batch_size, 1), y_batch = (batch_size,)
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
            log.critical("batch size was negative somehow", batch_size=batch_size)
        replace = batch_size > self.num_samples
        """
        get batch_size samples from self and replace once batch_size exceeds num_samples
        """
        choices = rng.choice(self.index, size=batch_size, replace=replace)
        x_batch = self.x[choices]
        y_batch = self.y[choices]
        log.debug("Selected training batch", batch_size=batch_size, replace=replace)
        return x_batch, y_batch

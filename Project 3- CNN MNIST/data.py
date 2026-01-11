from dataclasses import dataclass, InitVar, field
import numpy as np
import structlog
import tensorflow as tf

log = structlog.get_logger()

@dataclass
class MNIST:
    """generate spiral data"""
    rng: InitVar[np.random.Generator]
    train_ratio: float = 0.8
    
    # Data arrays (initialized post-construction)
    x_train: np.ndarray = field(init=False)
    x_val: np.ndarray = field(init=False)
    x_test: np.ndarray = field(init=False)
    y_train: np.ndarray = field(init=False)
    y_val: np.ndarray = field(init=False)
    y_test: np.ndarray = field(init=False)
    
    index: np.ndarray = field(init=False)
    
    def __post_init__(self, rng:np.random.Generator):
        # Load and normalize MNIST data
        (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = tf.keras.datasets.mnist.load_data()
        x_train_raw = self.normalize_and_expand(x_train_raw)
        x_test_raw = self.normalize_and_expand(x_test_raw)

        # Shuffle and split training data into train/val
        train_idx, val_idx = self.split_indices(len(y_train_raw), rng)

        self.x_train = x_train_raw[train_idx]
        self.y_train = y_train_raw[train_idx]
        self.x_val = x_train_raw[val_idx]
        self.y_val = y_train_raw[val_idx]
        self.x_test = x_test_raw
        self.y_test = y_test_raw
        self.index = np.arange(len(self.y_train))

    def normalize_and_expand(self, x: np.ndarray) -> np.ndarray:
        return np.expand_dims(x / 255.0, axis=-1)

    def split_indices(self, size: int, rng:np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        indices = np.arange(size)
        rng.shuffle(indices)
        split = int(self.train_ratio * size)
        return indices[:split], indices[split:]

    def get_batch(self, rng:np.random.Generator, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """get random batch"""
        choices = rng.choice(self.index, size=batch_size)
        return self.x_train[choices], self.y_train[choices] #x_train has dimensions 28x28x1
    
    def get_validation(self):
        return self.x_val, self.y_val
    
    def get_test(self):
        return self.x_test, self.y_test
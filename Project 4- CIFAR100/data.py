from dataclasses import dataclass, InitVar, field
import numpy as np
import structlog
import tensorflow as tf

log = structlog.get_logger()

@dataclass
class cifar10:
    """retreive data. also i know that my class name is misleading when im training cifar100
    kinda too lazy to change it."""
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
        # Load and normalize cifar10 data
        (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = tf.keras.datasets.cifar10.load_data()
        
        # convert to float32 in [0,1]
        x_train_raw = x_train_raw.astype(np.float32) / 255.0
        x_test_raw = x_test_raw.astype(np.float32) / 255.0

        # labels come as shape (N,1) — flatten to (N,)
        y_train_raw = y_train_raw.reshape(-1)
        y_test_raw = y_test_raw.reshape(-1)

        # shuffle and split training into train/val
        train_idx, val_idx = self.split_indices(len(y_train_raw), rng)

        self.x_train = x_train_raw[train_idx]
        self.y_train = y_train_raw[train_idx]
        self.x_val = x_train_raw[val_idx]
        self.y_val = y_train_raw[val_idx]
        self.x_test = x_test_raw
        self.y_test = y_test_raw
        self.index = np.arange(len(self.y_train))

    def split_indices(self, size: int, rng: np.random.Generator):
        indices = np.arange(size)
        rng.shuffle(indices)
        split = int(self.train_ratio * size)
        return indices[:split], indices[split:]

    def get_batch(self, rng: np.random.Generator, batch_size: int):
        choices = rng.choice(self.index, size=batch_size)
        return self.x_train[choices], self.y_train[choices]

    def get_validation(self):
        return self.x_val, self.y_val

    def get_test(self):
        return self.x_test, self.y_test
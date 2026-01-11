import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from typing import Union, Tuple, List
import structlog
import math

log = structlog.get_logger()

class Convolution(nnx.Module):
    """A single convolutional layer with dropout and optional L2 regularization."""

    def __init__(
        self,
        *,
        keys,
        L2_weight: float,
        dropout_rate: float,
        in_features: int,
        out_features: int,
        kernel_size: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]],
        padding: str = "SAME",
    ):
        self.rngs = nnx.Rngs(params=keys, dropout=keys)
        self.L2_weight = L2_weight
        self.padding = padding

        self.layer = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=self.padding,
            rngs=self.rngs,
        )
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=self.rngs)

    def __call__(self, x: jnp.ndarray, train: bool):
        x = self.layer(x)
        x = self.dropout(x, deterministic=not train)
        return x

    def L2_loss(self) -> jnp.ndarray:
        return jnp.sum(jnp.square(self.layer.kernel.value)) * self.L2_weight


class Classify(nnx.Module):
    """CNN classifier for MNIST."""

    def __init__(
        self,
        *,
        input_depth: int,
        layer_depths: List[int],
        layer_kernel_sizes: List[Tuple[int, int]],
        strides: List[Union[int, Tuple[int, int]]],
        num_classes: int,
        dropout_rate: float,
        L2_weight: float,
        rngs: nnx.Rngs,
        shape: List[int], 
    ):
        keys = rngs.params()

        self.input_depth = input_depth
        self.layer_depths = layer_depths
        self.layer_kernel_sizes = layer_kernel_sizes
        self.strides = strides
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.L2_weight = L2_weight

        # Depth of inputs to each layer
        input_features = [input_depth] + layer_depths[:-1]
        layer_keys = jax.random.split(keys, len(layer_depths))

        log.debug(
            "conv layers dimentinos:",
            layer_depthsdepths=layer_depths,
            input_features=input_features,
            layer_kernel_sizes=layer_kernel_sizes,
            strides=strides,
        )

        # Build convolutional layers
        self.layers = [
            Convolution(
                in_features=input_features[i],
                out_features=layer_depths[i],
                kernel_size=layer_kernel_sizes[i],
                strides=strides[i],
                keys=layer_keys[i],
                L2_weight=L2_weight,
                dropout_rate=dropout_rate,
            )
            for i in range(len(layer_depths))
        ]

        #flatten and account for downsampling caused by larger stride
        # normalize strides to a (n, 2) array of ints
        stride_arr = jnp.array(
            [[s, s] if isinstance(s, (int, np.integer)) else s for s in strides],
            dtype=int
        )
        # product of strides along each dimension
        stride_prod = jnp.prod(stride_arr, axis=0)   # shape (2,whatever)
        h, w = shape
        h = math.ceil(h / stride_prod[0])
        w = math.ceil(w / stride_prod[1])
        log.debug("h and w:", h = h, w =w)

        flatten_size = int(h * w * layer_depths[-1])

        self.final_layer = nnx.Linear(
            in_features=flatten_size,
            out_features=num_classes,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        val = x
        for layer in self.layers:
            val = layer(val, train)
            val = jax.nn.leaky_relu(val)

        val = val.reshape((val.shape[0], -1))  # (batch, h*w*d)
        return self.final_layer(val)

    def L2_loss(self) -> jnp.ndarray:
        return sum(layer.L2_loss() for layer in self.layers)

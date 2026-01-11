import jax
import jax.numpy as jnp
from flax import nnx
from typing import Union, Tuple, List
import structlog
import math

from .config import ModelSettings

settings = ModelSettings()

log = structlog.get_logger()

#Pooling
class Pooling(nnx.Module):
    def __init__(self, *, window=(2,2), strides=(2,2), padding="SAME"):
        self.window = window
        self.strides = strides
        self.padding = padding

    def __call__(self, x, train=False):
        return jax.lax.reduce_window(
            x,
            -jnp.inf,
            jax.lax.max,
            window_dimensions=(1, self.window[0], self.window[1], 1),
            window_strides=(1, self.strides[0], self.strides[1], 1),
            padding=self.padding,
        )

# GroupNorm
class GroupNorm(nnx.Module):
    """
    Group normalization with learnable affine parameters per channel.
    """
    def __init__(self, *, num_channels: int, num_groups: int = 8, epsilon: float = 1e-5):
        # store simple config
        self.num_channels = int(num_channels)
        self.num_groups = int(num_groups)
        self.epsilon = float(epsilon)

        # choose group count that divides channels (fallback by decrementing)
        G = min(self.num_groups, self.num_channels)
        while G > 1 and (self.num_channels % G) != 0:
            G -= 1
        self._G = max(1, G)

        # learnable affine parameters (per-channel)
        # nnx.Param wraps a leaf parameter; we initialize to gamma=1, beta=0
        self.scale = nnx.Param(jnp.ones((self.num_channels,), dtype=jnp.float32))
        self.bias = nnx.Param(jnp.zeros((self.num_channels,), dtype=jnp.float32))

    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        # x: (N, H, W, C)
        N, H, W, C = x.shape
        assert C == self.num_channels, "GroupNorm: channel mismatch"
        
        # reshape into groups: (N, H, W, G, C_per_group
        Cg = C // self._G
        xg = x.reshape((N, H, W, self._G, Cg))

        # compute mean/var over (H, W, Cperg)
        mean = jnp.mean(xg, axis=(1, 2, 4), keepdims=True)
        var = jnp.var(xg, axis=(1, 2, 4), keepdims=True)

        xg = (xg - mean) / jnp.sqrt(var + self.epsilon)
        x_norm = xg.reshape((N, H, W, C))

        # apply affine per-channel (broadcast over N,H,W)
        gamma = self.scale.value if hasattr(self.scale, "value") else self.scale
        beta = self.bias.value if hasattr(self.bias, "value") else self.bias
        gamma = gamma.reshape((1, 1, 1, C))
        beta = beta.reshape((1, 1, 1, C))

        return x_norm * gamma + beta

#Convolution
class Convolution(nnx.Module):
    """
    Single convolution with L2
    """
    def __init__(
        self,
        *,
        keys,
        L2_weight: float,
        in_features: int,
        out_features: int,
        kernel_size: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]],
        padding: str = "SAME",
    ):
        # store config
        self.L2_weight = float(L2_weight)
        self.padding = padding

        # normalize kernel and strides
        if isinstance(kernel_size, int):
            kh = kw = int(kernel_size)
        else:
            kh, kw = map(int, kernel_size)

        if isinstance(strides, int):
            sh = sw = int(strides)
        else:
            sh, sw = strides

        self.kernel_size = (kh, kw)
        self.strides = (sh, sw)

        # compute fan_in and He (Kaiming) std for ReLU
        fan_in = kh * kw * int(in_features)
        he_std = math.sqrt(2.0 / float(fan_in))

        # generate kernel
        key = keys if keys is not None else jax.random.PRNGKey(0)
        kernel_shape = (kh, kw, int(in_features), int(out_features))
        init_kernel = jax.random.normal(key, kernel_shape, dtype=jnp.float32) * jnp.array(he_std, dtype=jnp.float32)
        self.kernel = nnx.Param(init_kernel)

    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """
        x expected shape: (N, H, W, C) (NHWC).
        Returns NHWC.
        """
        kernel_arr = self.kernel.value if hasattr(self.kernel, "value") else self.kernel

        out = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=kernel_arr,
            window_strides=self.strides,
            padding=self.padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        return out

    def L2_loss(self) -> jnp.ndarray:
        k = self.kernel.value if hasattr(self.kernel, "value") else self.kernel
        return jnp.sum(jnp.square(k)) * self.L2_weight


class GaussianNoise(nnx.Module):
    def __init__(self, *, std: float, key):
        self.std = std
        self.rngs = nnx.Rngs(noise=key)

    def __call__(self, x, train: bool = False):
        if not train or self.std <= 0:
            return x
        noise = jax.random.normal(self.rngs.noise(), x.shape) * self.std
        return x + noise

# ResidualBlock
class ResidualBlock(nnx.Module):
    """
    Basic residual block: Conv -> GroupNorm -> ReLU -> Conv -> GroupNorm -> skip -> ReLU
    """
    def __init__(
        self,
        *,
        keys,# tuple of three keys: (conv1_key, conv2_key, skip_proj_key)
        L2_weight: float,
        in_features: int,
        out_features: int,
        kernel_size: Tuple[int, int] = (3, 3),
        strides: Tuple[int, int] = (1, 1),
    ):
        conv1_key, conv2_key, proj_key = keys

        # conv1: in_features -> out_features
        self.conv1 = Convolution(
            keys=conv1_key,
            L2_weight=L2_weight,
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
        )
        self.norm1 = GroupNorm(num_channels=out_features)

        # conv2: out_features -> out_features
        self.conv2 = Convolution(
            keys=conv2_key,
            L2_weight=L2_weight,
            in_features=out_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=(1, 1),
        )
        self.norm2 = GroupNorm(num_channels=out_features)

        # projection on skip if needed
        if in_features != out_features or strides != (1, 1):
            self.proj = nnx.Conv(
                in_features=in_features,
                out_features=out_features,
                kernel_size=(1, 1),
                strides=strides,
                padding="SAME",
                rngs=nnx.Rngs(params=proj_key),
            )
        else:
            self.proj = None

        self.noise = GaussianNoise(std=settings.noise_std, key = proj_key)

    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        identity = x

        out = self.conv1(x, train)
        out = self.norm1(out, train)
        out = jax.nn.relu(out)
        
        if train:
            out = self.noise(out,train)

        out = self.conv2(out, train)
        out = self.norm2(out, train)

        if self.proj is not None:
            identity = self.proj(identity)

        out = out + identity
        return jax.nn.relu(out)

    def L2_loss(self) -> jnp.ndarray:
        loss = self.conv1.L2_loss() + self.conv2.L2_loss()
        if self.proj is not None:
            loss += jnp.sum(jnp.square(self.proj.kernel.value))
        return loss

# classify
class Classify(nnx.Module):
    """
    CNN classifier built from ResidualBlocks.
    """
    def __init__(
        self,
        *,
        input_depth: int,
        layer_depths: List[int],
        layer_kernel_sizes: List[Tuple[int, int]],
        strides: List[Union[int, Tuple[int,int]]],
        num_classes: int,
        L2_weight: float,
        rngs: nnx.Rngs,
        shape: List[int],
    ):
        keys = rngs.params()             # root key (jax.Array)

        input_features = [input_depth] + layer_depths[:-1]
        layers = []

        # We'll need 3 keys per residual block plus one key for final linear:
        total_keys = len(layer_depths) * 3 + 1
        all_keys = jax.random.split(keys, total_keys)
        block_keys = all_keys[:-1]
        final_key = all_keys[-1]

        for i, out_ch in enumerate(layer_depths):
            k1, k2, k3 = block_keys[3*i:3*i+3]
            stride_val = (strides[i], strides[i]) if isinstance(strides[i], int) else strides[i]
            block = ResidualBlock(
                keys=(k1, k2, k3),
                L2_weight=L2_weight,
                in_features=input_features[i],
                out_features=out_ch,
                kernel_size=tuple(layer_kernel_sizes[i]),
                strides=stride_val,
            )
            layers.append(block)

            # pooling between blocks (except after last)
            if i < len(layer_depths) - 1:
                layers.append(Pooling(window=(2,2), strides=(2,2)))

        self.layers = nnx.List(layers)

        # compute flatten size after downsampling
        h, w = shape[0], shape[1]
        for s in strides:
            sh, sw = (s, s) if isinstance(s, int) else s
            h = math.ceil(h / sh)
            w = math.ceil(w / sw)
        # account for pooling between blocks
        h = h // (2 ** (len(layer_depths) - 1))
        w = w // (2 ** (len(layer_depths) - 1))
        flatten_size = h * w * layer_depths[-1]

        # final linear
        self.final_layer = nnx.Linear(
            in_features=flatten_size,
            out_features=num_classes,
            rngs=nnx.Rngs(params=final_key),
        )

    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        out = x
        for layer in self.layers:
            # pooling and blocks all accept (x, train) signature
            out = layer(out, train)
        out = out.reshape((out.shape[0], -1))
        return self.final_layer(out)

    def L2_loss(self) -> jnp.ndarray:
        return sum(layer.L2_loss() for layer in self.layers if hasattr(layer, "L2_loss"))

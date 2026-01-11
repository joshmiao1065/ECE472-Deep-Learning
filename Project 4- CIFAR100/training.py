import jax.numpy as jnp
import jax
import numpy as np
import optax
import structlog
import orbax.checkpoint as ocp
from typing import Optional
from pathlib import Path

from flax import nnx
from tqdm import trange
from .config import TrainingSettings
from .data import cifar10
from .model import Classify

log = structlog.get_logger() 

@nnx.jit
def train_step_cifar10(
    model: Classify,
    optimizer: nnx.Optimizer,
    x: jnp.ndarray,
    y: jnp.ndarray,
 ):
    
    """perform on training step"""

    # x shape: (batch_size, 32, 32, 3); y shape: (batch_size,)
    def loss_function(model: Classify):
        logits = model(x, True)
        ce_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
        L2_loss = model.L2_loss()

        return ce_loss + L2_loss
    
    loss, grads = nnx.value_and_grad(loss_function)(model)
    optimizer.update(model, grads)
    return loss

def augment(
    key: jnp.array = jax.random.PRNGKey(472),
    images: jnp.array = jnp.ndarray,
    *,
    p_hflip: float = 0.35, #probabilities for performing these augmentations
    pad: int = 4,
    brightness: float = 0.2,#changes in brightness, conttrast, and saturation
    contrast: float = 0.2,
    saturation: float = 0.2,
    p_cutout: float = 0.35,
    # i define this in the cutout function bc i break vmap otherwise and im too lazy to figure out why: max_cutout: int = 8,
    p_greyscale: float = 0.3,   
    p_noise: float = 0.85,       
    noise_std: float = 0.15,    # stddev of additive Gaussian noise
) -> jnp.ndarray:
    """
    Batch augmentation
    images: float32 in [0,1], shape (B,H,W,C).
    Returns augmented images, same shape and dtype.
    """
    B, H, W, C = images.shape

    # Split top-level key into independent subkeys.
    # We need separate keys for flips, crop, color, cutout, greyscale, noise_mask, noise_rng.
    key, k_flip_h, k_crop, k_color, k_cutout, k_grey, k_noise_mask, k_noise = (
        jax.random.split(key, 8)
    )

    """i got rid of rotation because many classes in the cifar datasets dont make sense to flip and rotating
    it by a small amount instead didnt really help much. also it probably didnt help that i couldnt find a roation funciton in the
    jax documentation lol"""

    # Horizontal flip
    hmask = jax.random.bernoulli(k_flip_h, p_hflip, (B,))
    images = jnp.where(hmask[:, None, None, None], images[:, :, ::-1, :], images)

    """i also axed vertical flip because its genunely just not worth it given that many of the classes in both cifar datasets
    #wouldnt realistically have many vertically flipped images"""

    # Crop (random crop from reflected padding)
    padded = jnp.pad(images, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect")
    max_x = padded.shape[1] - H
    max_y = padded.shape[2] - W
    k_crop_x, k_crop_y = jax.random.split(k_crop, 2)
    xs = jax.random.randint(k_crop_x, (B,), 0, max_x + 1)
    ys = jax.random.randint(k_crop_y, (B,), 0, max_y + 1)

    def crop_one(img, x, y):
        return jax.lax.dynamic_slice(img, (x, y, 0), (H, W, C))

    images = jax.vmap(crop_one)(padded, xs, ys)

    # Color jitter (brightness / contrast / saturation)
    k_b, k_c, k_s = jax.random.split(k_color, 3)
    bright_f = jax.random.uniform(k_b, (B,), minval=-brightness, maxval=brightness)
    contrast_f = jax.random.uniform(k_c, (B,), minval=1.0 - contrast, maxval=1.0 + contrast)
    sat_f = jax.random.uniform(k_s, (B,), minval=-saturation, maxval=saturation)

    def apply_color(bf, cf, sf, img):
        # brightness
        img = img + bf
        # contrast
        mean = jnp.mean(img, axis=(0, 1), keepdims=True)
        img = (img - mean) * cf[..., None, None] + mean
        # saturation (convert to luminance and lerp)
        lum = img[..., 0] * 0.2989 + img[..., 1] * 0.5870 + img[..., 2] * 0.1140
        lum = lum[..., None]
        img = img * (1.0 - sf)[..., None, None] + lum * sf[..., None, None]
        return jnp.clip(img, 0.0, 1.0)

    images = jax.vmap(apply_color)(bright_f, contrast_f, sat_f, images)

    # Cutout
    cut_mask = jax.random.bernoulli(k_cutout, p_cutout, (B,))  # which images get cutout
    # create 4 keys per image (one for cut_h, cut_w, ky, kx)
    per_cut_keys = jax.random.split(k_cutout, B * 4).reshape((B, 4, 2))

    def do_cutout(karr, do, img, max_cutout = 8):
        Hc, Wc, Cc = img.shape

        def true_fn(args):
            karr, img = args
            k0, k1, k2, k3 = karr

            # pick random cutout size
            cut_h = jax.random.randint(k0, (), 1, max_cutout + 1)
            cut_w = jax.random.randint(k1, (), 1, max_cutout + 1)

            # pick random top-left corner
            ky = jax.random.randint(k2, (), 0, Hc - cut_h + 1)
            kx = jax.random.randint(k3, (), 0, Wc - cut_w + 1)

            # create a mask of ones (static shape)
            mask = jnp.ones((Hc, Wc, Cc), dtype=img.dtype)

            # create indices for cutout
            rows = jnp.arange(Hc)
            cols = jnp.arange(Wc)
            row_mask = (rows < ky) | (rows >= ky + cut_h)
            col_mask = (cols < kx) | (cols >= kx + cut_w)

            # broadcast row/col masks
            cut_mask_local = jnp.outer(row_mask, col_mask).astype(img.dtype)
            cut_mask_local = cut_mask_local[..., None]  # expand to (H, W, 1)
            mask = mask * cut_mask_local           # apply mask

            return img * mask

        def false_fn(args):
            return args[1]

        return jax.lax.cond(do, true_fn, false_fn, (karr, img))

    images = jax.vmap(do_cutout)(per_cut_keys, cut_mask, images)

    #greyscale
    if p_greyscale > 0.0:
        grey_mask = jax.random.bernoulli(k_grey, p_greyscale, (B,))  # which images -> greyscale
        # compute luminance and stack to 3 channels for those images
        lum = images[..., 0] * 0.2989 + images[..., 1] * 0.5870 + images[..., 2] * 0.1140  # (B,H,W)
        lum = lum[..., None]  # (B,H,W,1)
        grey_images = jnp.concatenate([lum, lum, lum], axis=-1)  # (B,H,W,3)
        images = jnp.where(grey_mask[:, None, None, None], grey_images, images)

    #awgn
    if p_noise > 0.0 and noise_std > 0.0:
        noise_mask = jax.random.bernoulli(k_noise_mask, p_noise, (B,))
        noise = jax.random.normal(k_noise, (B, H, W, C)) * noise_std
        noisy = images + noise
        noisy = jnp.clip(noisy, 0.0, 1.0)
        images = jnp.where(noise_mask[:, None, None, None], noisy, images)

    return jnp.clip(images, 0.0, 1.0)

def cifar10_train(
    model: Classify,
    optimizer: nnx.Optimizer,
    data: cifar10,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
) -> None:
    """train with grad descend"""
    log.info("started training")

    # create a persistent jnp PRNGKey seeded from the numpy RNG you already have
    jnp_key = jax.random.PRNGKey(int(np_rng.integers(0, 2 ** 31 - 1)))

    bar = trange(settings.num_iters)
    for i in bar:
        x_np, y_np = data.get_batch(np_rng, settings.batch_size)
        x = jnp.asarray(x_np, dtype=jnp.float32)
        y = jnp.asarray(y_np, dtype=jnp.int32).reshape(-1)

        # get subkey and augment the batch (augmentation is outside the jitted train step)
        jnp_key, subkey = jax.random.split(jnp_key)
        x_aug = augment(subkey, x)

        loss = train_step_cifar10(model, optimizer, x_aug, y)

        bar.set_description(f"loss at {i} => {loss:.6f}")
        bar.refresh()
    
    log.info("done training")

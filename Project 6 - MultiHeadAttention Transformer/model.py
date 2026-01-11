import structlog
import jax
import jax.numpy as jnp
import flax.linen as nnx
from typing import Optional, Tuple
from .config import ModelSettings

log = structlog.get_logger()

Array = jnp.ndarray

class MultiHeadAttention(nnx.Module):
    """Multi-head self-attention (1-D sequence)
    Returns (out, attn_weights) when return_attn=True to aid testing.
    """
    d_model: int
    num_heads: int
    dropout_rate: float = 0.0
    use_bias: bool = True

    @nnx.compact
    def __call__(self,
                 x: Array,
                 deterministic: bool = True,
                 return_attn: bool = False
                 ) -> Tuple[Array, Optional[Array]]:
        """
        x: (batch, seq_len, d_model)
        """
        B, T, D = x.shape
        assert D == self.d_model, "input embedding dim must match d_model"
        assert D % self.num_heads == 0, "d_model must be divisible by num_heads"
        head_dim = D // self.num_heads

        log.debug("MHA.call", batch=B, seq_len=T, d_model=D, num_heads=self.num_heads, head_dim=head_dim)

        # Combined linear for qkv: (B, T, 3*D)
        qkv = nnx.Dense(3 * D, use_bias=self.use_bias, name="qkv_proj")(x)
        # split and reshape => (B, num_heads, T, head_dim)
        qkv = qkv.reshape(B, T, 3, self.num_heads, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, H, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # compute attention logits: (B, H, T, T)
        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k)
        logits = logits * (1.0 / jnp.sqrt(head_dim))

        attn = jax.nn.softmax(logits, axis=-1)
        attn = nnx.Dropout(rate=self.dropout_rate)(attn, deterministic=deterministic)

        out_per_head = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        out_per_head = jnp.transpose(out_per_head, (0, 2, 1, 3))  # (B, T, H, head_dim)
        out = out_per_head.reshape(B, T, D)

        out = nnx.Dense(D, use_bias=self.use_bias, name="out_proj")(out)
        out = nnx.Dropout(rate=self.dropout_rate)(out, deterministic=deterministic)

        log.debug("MHA.output", out_shape=out.shape)

        if return_attn: # attn shape: (B, H, T, T)
            return out, attn
        return out, None


class TransformerBlock(nnx.Module):
    """Single Transformer block with pre-LN, MHA, feed-forward, residuals.
    Designed for 1-D sequences. LayerNorm before sublayer
    """
    d_model: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.0
    pre_norm: bool = True

    @nnx.compact
    def __call__(self,
                 x: Array,
                 deterministic: bool = True,
                 return_attn: bool = False
                 ) -> Tuple[Array, Optional[Array]]:
        """
        x: (B, T, d_model)
        returns (out, attn_weights_if_requested)
        """
        log.debug("TransformerBlock.call", x_shape=x.shape, pre_norm=self.pre_norm)
        
        #MHA sublayer
        if self.pre_norm:
            y = nnx.LayerNorm(name="ln_1")(x)
        else:
            y = x

        mha = MultiHeadAttention(self.d_model, self.num_heads, dropout_rate=self.dropout_rate, name="mha")
        attn_out, attn_weights = mha(y, deterministic=deterministic, return_attn=return_attn)
        x = x + attn_out
        log.debug("TransformerBlock.after_mha", x_shape=x.shape)
        
        #Feed-forward sublayer
        if self.pre_norm:
            y = nnx.LayerNorm(name="ln_2")(x)
        else:
            y = x

        y = nnx.Dense(self.ff_dim, name="ff_1")(y)
        y = nnx.relu(y)
        y = nnx.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        y = nnx.Dense(self.d_model, name="ff_2")(y)
        y = nnx.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)

        out = x + y
        log.debug("TransformerBlock.output", out_shape=out.shape)
        return out, (attn_weights if return_attn else None)

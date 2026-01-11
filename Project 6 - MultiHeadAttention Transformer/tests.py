# tests.py
import structlog
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from typing import List, Tuple
from .model import MultiHeadAttention, TransformerBlock

log = structlog.get_logger()
KEY = random.PRNGKey(427)


def _init_module(mod, x, rng=KEY):
    return mod.init(rng, x, deterministic=True)


def test_mha_shape_and_attn_sums() -> Tuple[bool, str]:
    """Shape checks and attention rows sum to 1 after softmax."""
    B, T, D = 2, 8, 32
    H = 4
    x = random.normal(KEY, (B, T, D))

    mha = MultiHeadAttention(d_model=D, num_heads=H, dropout_rate=0.0)
    params = _init_module(mha, x)

    out, attn = mha.apply(params, x, deterministic=True, return_attn=True)
    
    assert out.shape == (B, T, D)
    assert attn.shape == (B, H, T, T)
    # each query distribution sums to 1
    row_sums = jnp.sum(attn, axis=-1)
    assert jnp.allclose(row_sums, jnp.ones_like(row_sums), atol=1e-5)
    return True, "passed"

def test_mha_deterministic_behavior() -> Tuple[bool, str]:
    """Fixed PRNG/ deterministic=True yields identical outputs across calls."""
    B, T, D = 2, 6, 16
    H = 2
    x = random.normal(KEY, (B, T, D))
    mha = MultiHeadAttention(d_model=D, num_heads=H, dropout_rate=0.0)
    params = _init_module(mha, x)

    out1, _ = mha.apply(params, x, deterministic=True, return_attn=False)
    out2, _ = mha.apply(params, x, deterministic=True, return_attn=False)

    assert jnp.allclose(out1, out2), "outputs differ despite deterministic=True"
    return True, "passed"

def test_permutation_equivariance_mha() -> Tuple[bool, str]:
    """Without positional encodings, MHA should be permutation equivariant:
       permuting input positions permutes outputs the same way.
    """
    B, T, D = 2, 7, 24
    H = 3
    x = random.normal(KEY, (B, T, D))
    perm = jax.random.permutation(KEY, T)

    mha = MultiHeadAttention(d_model=D, num_heads=H, dropout_rate=0.0)
    params = _init_module(mha, x)

    out, _ = mha.apply(params, x, deterministic=True, return_attn=False)
    out_perm, _ = mha.apply(params, x[:, perm, :], deterministic=True, return_attn=False)
    
    # out_perm should equal out with positions permuted
    assert jnp.allclose(out[:, perm, :], out_perm, atol=1e-6)
    return True, "passed"

def test_transformerblock_grad_flow_basic() -> Tuple[bool, str]:
    """Quick gradient sanity check: gradient of sum(out) wrt inputs is nonzero."""

    B, T, D = 1, 5, 16
    x = random.normal(KEY, (B, T, D))

    block = TransformerBlock(d_model=D, num_heads=4, ff_dim=4 * D, dropout_rate=0.0)
    params = block.init(KEY, x, deterministic=True)

    def loss_fn(inp):
        out, _ = block.apply(params, inp, deterministic=True, return_attn=False)
        return jnp.sum(out)

    g = jax.grad(loss_fn)(x)
    # gradient should have same shape and not be all zeros
    assert g.shape == x.shape
    assert jnp.any(jnp.abs(g) > 0.0)
    return True, "passed"

def _manual_single_head_forward_from_params(params, x):
    """Given params from MultiHeadAttention with num_heads=1, compute manual forward."""
    log.debug("manual_single_head_forward start", x_shape=x.shape)
    p = params['params']
    qkv_k = p['qkv_proj']['kernel']
    qkv_b = p['qkv_proj'].get('bias', None)
    out_k = p['out_proj']['kernel']
    out_b = p['out_proj'].get('bias', None)

    B, T, D = x.shape
    H = 1
    head_dim = D // H

    qkv = jnp.einsum("btd,df->btf", x, qkv_k) #big thanks to mr. eric eng for showing me einstein summations
    if qkv_b is not None:
        qkv += qkv_b
    qkv = qkv.reshape(B, T, 3, H, head_dim)
    qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
    q, k, v = qkv[0], qkv[1], qkv[2]

    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(head_dim)
    attn = jax.nn.softmax(logits, axis=-1)
    out_per_head = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
    out_per_head = jnp.transpose(out_per_head, (0, 2, 1, 3))
    out_pre = out_per_head.reshape(B, T, D)

    out = jnp.einsum("btd,df->btf", out_pre, out_k)
    if out_b is not None:
        out += out_b

    log.debug("manual_single_head_forward.end", out_shape=out.shape)
    return out, attn, out_pre


def test_head_collapse_single_vs_manual() -> Tuple[bool, str]:
    """When num_heads=1, module should equal manual single-head computation (outputs and grads)."""
    #not liskov sub exactly but i think this achieves the same test
    log.debug("test_head_collapse start")
    B, T, D = 2, 6, 16
    x = random.normal(KEY, (B, T, D))

    mha = MultiHeadAttention(d_model=D, num_heads=1, dropout_rate=0.0)
    params = _init_module(mha, x)

    out_mod, _ = mha.apply(params, x, deterministic=True, return_attn=True)
    out_manual, _, _ = _manual_single_head_forward_from_params(params, x)

    if not jnp.allclose(out_mod, out_manual, atol=1e-6):
        return False, "head-collapse forward mismatch"

    def loss_mod(inp): return jnp.sum(mha.apply(params, inp, deterministic=True)[0])
    def loss_manual(inp): return jnp.sum(_manual_single_head_forward_from_params(params, inp)[0])

    g_mod, g_manual = jax.grad(loss_mod)(x), jax.grad(loss_manual)(x)

    if not jnp.allclose(g_mod, g_manual, atol=1e-5):
        return False, "head-collapse input gradients mismatch"

    log.debug("test_head_collapse passed")
    return True, "passed"


def test_softmax_stability_scaling() -> Tuple[bool, str]:
    """Ensure softmax doesn't blow up with very large logits (scaling in place)."""
    log.debug("test_softmax_stability start")
    B, T, D, H = 1, 4, 32, 4
    x = random.normal(KEY, (B, T, D)) * 1e3
    mha = MultiHeadAttention(d_model=D, num_heads=H, dropout_rate=0.0)
    params = _init_module(mha, x)

    out, attn = mha.apply(params, x, deterministic=True, return_attn=True)
    if not (jnp.all(jnp.isfinite(out)) and jnp.all(jnp.isfinite(attn))):
        return False, "non-finite values in output or attention"

    log.debug("test_softmax_stability passed")
    return True, "passed"


def test_head_additivity_and_reconstruction() -> Tuple[bool, str]:
    """Reconstruct final output from per-head outputs + out_proj and compare to module output."""
    log.debug("test_head_additivity started")
    B, T, D, H = 2, 5, 32, 4
    x = random.normal(KEY, (B, T, D))
    mha = MultiHeadAttention(d_model=D, num_heads=H, dropout_rate=0.0)
    params = _init_module(mha, x)
    out_mod, _ = mha.apply(params, x, deterministic=True, return_attn=True)

    p = params['params']
    qkv_k, qkv_b = p['qkv_proj']['kernel'], p['qkv_proj'].get('bias', None)
    out_k, out_b = p['out_proj']['kernel'], p['out_proj'].get('bias', None)

    qkv = jnp.einsum("btd,df->btf", x, qkv_k)
    if qkv_b is not None:
        qkv += qkv_b
    head_dim = D // H
    qkv = qkv.reshape(B, T, 3, H, head_dim).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(head_dim)
    attn = jax.nn.softmax(logits, axis=-1)
    out_per_head = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
    pre_concat = out_per_head.transpose(0, 2, 1, 3).reshape(B, T, D)

    recon = jnp.einsum("btd,df->btf", pre_concat, out_k)
    if out_b is not None:
        recon += out_b

    if not jnp.allclose(recon, out_mod, atol=1e-6):
        return False, "reconstructed output != module output"

    log.debug("test_head_additivity passed")
    return True, "passed"


def test_zero_out_head_and_contribution() -> Tuple[bool, str]:
    """Zero one head's qkv params; diff = contribution of that head."""
    log.debug("test_zero_out_head start")
    from jax import tree_util

    B, T, D, H, head_to_zero = 2, 5, 32, 4, 1
    x = random.normal(KEY, (B, T, D))
    mha = MultiHeadAttention(d_model=D, num_heads=H, dropout_rate=0.0)
    params = _init_module(mha, x)
    out_orig, _ = mha.apply(params, x, deterministic=True, return_attn=True)

    p = params['params']
    qkv_k, qkv_b = p['qkv_proj']['kernel'], p['qkv_proj'].get('bias', None)
    out_k, out_b = p['out_proj']['kernel'], p['out_proj'].get('bias', None)

    head_dim = D // H
    qkv = jnp.einsum("btd,df->btf", x, qkv_k)
    if qkv_b is not None:
        qkv += qkv_b
    qkv = qkv.reshape(B, T, 3, H, head_dim).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(head_dim)
    attn = jax.nn.softmax(logits, axis=-1)
    out_per_head = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
    out_per_head_t = out_per_head.transpose(0, 2, 1, 3)

    # compute contribution of that head
    mask = jnp.arange(H) == head_to_zero
    pre_concat_head = (out_per_head_t * mask[None, None, :, None]).reshape(B, T, D)
    head_contrib = jnp.einsum("btd,df->btf", pre_concat_head, out_k)

    # zero head params
    params_zeroed = tree_util.tree_map(lambda x: x.copy() if hasattr(x, 'copy') else x, params)
    def zero_head_qkv(kern, h):
        kern = kern.at[:, h*head_dim:(h+1)*head_dim].set(0.0)
        kern = kern.at[:, D + h*head_dim:D + (h+1)*head_dim].set(0.0)
        kern = kern.at[:, 2*D + h*head_dim:2*D + (h+1)*head_dim].set(0.0)
        return kern
    qkv_kz = zero_head_qkv(qkv_k, head_to_zero)
    params_zeroed['params']['qkv_proj']['kernel'] = qkv_kz
    if qkv_b is not None:
        b = qkv_b.at[head_to_zero*head_dim:(head_to_zero+1)*head_dim].set(0.0)
        b = b.at[D + head_to_zero*head_dim:D + (head_to_zero+1)*head_dim].set(0.0)
        b = b.at[2*D + head_to_zero*head_dim:2*D + (head_to_zero+1)*head_dim].set(0.0)
        params_zeroed['params']['qkv_proj']['bias'] = b

    out_zeroed, _ = mha.apply(params_zeroed, x, deterministic=True, return_attn=True)
    diff = out_orig - out_zeroed

    if not jnp.allclose(diff, head_contrib, atol=1e-5):
        return False, "zero-out-head diff != head contribution"

    log.debug("test_zero_out_head passed")
    return True, "passed"


def test_divisibility_check_error() -> Tuple[bool, str]:
    """d_model % num_heads != 0 should assert / error during init or call."""
    log.debug("test_divisibility_check start")
    B, T, D, H = 1, 4, 30, 4
    x = random.normal(KEY, (B, T, D))
    try:
        mha = MultiHeadAttention(d_model=D, num_heads=H, dropout_rate=0.0)
        _ = _init_module(mha, x)
    except (AssertionError, ValueError) as e:
        log.debug("test_divisibility_check passed", err_type=type(e).__name__)
        return True, f"passed ({type(e).__name__})"
    return False, "expected error for non-divisible d_model but succeeded"


def test_single_token_sequence_T1() -> Tuple[bool, str]:
    """Ensure T=1 works and shapes are preserved."""
    log.debug("test_single_token_T1 start")
    B, T, D, H = 2, 1, 16, 4
    x = random.normal(KEY, (B, T, D))
    mha = MultiHeadAttention(d_model=D, num_heads=H, dropout_rate=0.0)
    params = _init_module(mha, x)
    out, attn = mha.apply(params, x, deterministic=True, return_attn=True)
    assert out.shape == (B, T, D)
    assert attn.shape == (B, H, T, T)
    assert jnp.all(jnp.isfinite(out))
    log.debug("test_single_token_T1 passed")
    return True, "passed"


def test_backprop_through_attention_weights() -> Tuple[bool, str]:
    """Gradient through attention: d loss / d x should be non-zero."""
    log.debug("test_backprop_through_attention start")
    B, T, D, H = 1, 6, 24, 4
    x = random.normal(KEY, (B, T, D))
    mha = MultiHeadAttention(d_model=D, num_heads=H, dropout_rate=0.0)
    params = _init_module(mha, x)

    def loss_fn(inp):
        out, _ = mha.apply(params, inp, deterministic=True, return_attn=True)
        return jnp.sum(out[:, 0, :])

    g = jax.grad(loss_fn)(x)
    if not jnp.any(jnp.abs(g) > 0.0):
        return False, "gradient through attention is zero"
    log.debug("test_backprop_through_attention passed")
    return True, "passed"


def run_tests() -> List[Tuple[str, bool, str]]:
    tests = [
        ("mha_shape_and_attn_sums", test_mha_shape_and_attn_sums),
        ("mha_deterministic", test_mha_deterministic_behavior),
        ("mha_permutation_equivariance", test_permutation_equivariance_mha),
        ("transformer_grad_flow", test_transformerblock_grad_flow_basic),
        ("head_collapse_single_vs_manual", test_head_collapse_single_vs_manual),
        ("softmax_stability_scaling", test_softmax_stability_scaling),
        ("head_additivity_and_reconstruction", test_head_additivity_and_reconstruction),
        ("zero_out_head_and_contribution", test_zero_out_head_and_contribution),
        ("divisibility_check_error", test_divisibility_check_error),
        ("single_token_T1", test_single_token_sequence_T1),
        ("backprop_through_attention", test_backprop_through_attention_weights),
    ]
    results = []
    for name, fn in tests:
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, f"exception: {e}"
        log.info("test_result", test=name, passed=ok, message=msg)
        results.append((name, ok, msg))
    return results

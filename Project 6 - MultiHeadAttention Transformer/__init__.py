import structlog
import jax
import numpy as np

from .logging import configure_logging
from .config import load_settings
from .model import TransformerBlock
from .tests import run_tests

log = structlog.get_logger()

def main() -> None:
    """CLI entry point."""
    configure_logging()
    settings = load_settings()
    log.info("Loaded settings", settings=settings.model.dict() if hasattr(settings, "model") else str(settings))

    # PRNG
    seed = int(getattr(settings, "random_seed", 427))
    key = jax.random.PRNGKey(seed)
    data_key, model_key = jax.random.split(key)

    # instantiate model
    cfg = settings.model
    d_model = int(cfg.d_model)
    num_heads = int(cfg.num_heads)
    dropout = float(getattr(cfg, "dropout_rate", 0.0))
    ff_dim = max(4 * d_model, getattr(cfg, "ff_dim", 4 * d_model))  # fallback

    log.info("building TransformerBlock", d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
    block = TransformerBlock(d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, dropout_rate=dropout)

    #tets time
    log.info("Running tests. im so nervous")
    results = run_tests()

    passed = all(r[1] for r in results)
    log.info("Test summary", passed=passed, results=results)

    if not passed:
        # print helpful failure summary
        failures = [(name, msg) for name, ok, msg in results if not ok]
        for name, msg in failures:
            log.error("Test failed bruh", test=name, reason=msg)
        raise SystemExit(1)
    else:
        log.info("All tests passed yippeeeeeeeeeeee")

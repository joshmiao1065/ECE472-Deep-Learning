import structlog

from .logging import configure_logging
from .cli import main as cli_main


def main() -> None:
    "CLI entry point"
    configure_logging()
    log = structlog.get_logger()
    log.info("Starting hw01")
    cli_main()

"""Shared logging setup.

Call configure_logging() once from an entrypoint (a script or the CLI).
Library modules just call logging.getLogger(__name__) and never configure
handlers themselves, so importing them never has side effects on logging.
"""

import logging
import sys

from config import LOG_LEVEL


def configure_logging(level=LOG_LEVEL):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

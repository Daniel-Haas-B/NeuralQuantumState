import logging
import os
import sys


def pytest_configure(config):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

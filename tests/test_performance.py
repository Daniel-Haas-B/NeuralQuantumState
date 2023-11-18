import logging
import subprocess
from time import time

import pytest


@pytest.fixture
def playground_path():
    # Replace this with the actual logic to determine the playground path
    # For example, it could be a hardcoded path, a command-line argument, or an environment variable
    return (
        "/Users/haas/Documents/Masters/GANQS/src/simulation_scripts/ffnn_playground.py"
    )


def execute_script(playground_path):
    try:
        result = subprocess.run(
            ["python", playground_path], capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return e.stderr


def test_runs(playground_path):
    logger = logging.getLogger(__name__)

    time1 = time()
    output1 = None
    try:
        for i in range(6):
            output1 = execute_script(playground_path)
    except Exception as e:
        logger.info(e)
    time1 = time() - time1

    time2 = time()
    output2 = None
    try:
        for i in range(6):
            output2 = execute_script(playground_path.replace(".py", "_fast.py"))
        time2 = time() - time2
    except Exception as e:
        logger.info(e)

    improv = (time1 - time2) / time1
    logger.info(f"Pct improvement: {improv*100:.2f}%")

    assert output1 == output2

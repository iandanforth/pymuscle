import sys
import os

import re
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)
import pymuscle


def test_version():
    assert re.match(r'\d\.\d\.\d', pymuscle.__version__)

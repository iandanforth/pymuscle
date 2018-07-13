import re
import pymuscle


def test_version():
    assert re.match(r'\d\.\d\.\d', pymuscle.__version__)

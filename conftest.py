"""Repository-wide pytest hooks.

``fast`` marker: the quick CPU subset (``pytest -m fast``).  A test is fast because it was *measured* to be:
``tests/fast_tests.json`` maps test ids to their CPU durations from one run of the whole suite, and this hook marks the
listed tests at collection.  Nothing is marked by hand; a test that is not in the file is not fast, whatever it looks like.
"""
import json
import os

import pytest

FAST_LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "fast_tests.json")


def pytest_collection_modifyitems(config, items):
    try:
        with open(FAST_LIST) as f:
            fast = json.load(f).get("tests", {})
    except (OSError, ValueError):
        return
    root = str(config.rootpath)
    for item in items:
        nodeid = item.nodeid
        if nodeid in fast or os.path.relpath(str(item.fspath), root).replace(os.sep, "/") + "::" + nodeid.split("::", 1)[-1] in fast:
            item.add_marker(pytest.mark.fast)

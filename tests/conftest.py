import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, os.path.join(ROOT, "preprocessing"), os.path.join(ROOT, "datasets")):
    if p not in sys.path:
        sys.path.insert(0, p)
import synth  # noqa: E402


@pytest.fixture(scope="session")
def ctx(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    name, truth, info = synth.build(str(d), frames=12)
    return {"root": str(d), "name": name, "truth": truth, "info": info, "split": "validation"}

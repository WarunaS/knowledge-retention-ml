"""Live check against the real Hugging Face datasets (network). Skipped by default.

    pytest -m network
"""

import pytest

from krs.collect import trajectories, wildchat
from krs.config import load_config

pytestmark = pytest.mark.network


def _rev(name):
    return load_config("collect")["datasets"][name]["revision"]


def test_wildchat_live():
    convs = list(wildchat.load(3, revision=_rev("wildchat")))
    assert len(convs) == 3 and all(c.turns for c in convs)


@pytest.mark.parametrize("name", ["swe_agent", "openhands"])
def test_trajectories_live(name):
    [c] = list(trajectories.load(name, 1, revision=_rev(name)))
    assert c.project_id and len(c.turns) > 2

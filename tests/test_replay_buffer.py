import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from utils.move_encoder import get_policy_vector_size  # noqa: E402
from utils.replay_buffer import StreamingReplayBuffer  # noqa: E402

# Skip if torch or webdataset not installed
torch = pytest.importorskip("torch")
pytest.importorskip("webdataset")


def test_replay_buffer_add_sample(tmp_path):
    buf = StreamingReplayBuffer(tmp_path)
    state = np.zeros((12, 8, 8), dtype=np.float32)
    policy = np.zeros(get_policy_vector_size(), dtype=np.float32)
    for v in [0.1, -0.2, 0.3]:
        buf.add(state, policy, v)
    assert len(buf) == 3
    states, policies, values = buf.sample(2)
    assert states.shape[0] == 2
    assert policies.shape == (2, policy.size)
    assert values.shape[0] == 2
    buf.close()

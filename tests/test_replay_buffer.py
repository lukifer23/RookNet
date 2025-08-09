import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import numpy as np
import pytest

# Skip if webdataset is not installed
pytest.importorskip("webdataset")

from utils.replay_buffer import StreamingReplayBuffer
from utils.move_encoder import get_policy_vector_size


def test_replay_buffer_add_sample(tmp_path):
    state = np.zeros((12, 8, 8), dtype=np.float32)
    policy = np.zeros(get_policy_vector_size(), dtype=np.float32)
    with StreamingReplayBuffer(tmp_path) as buf:
        for v in [0.1, -0.2, 0.3]:
            buf.add(state, policy, v)
        assert len(buf) == 3
        torch = pytest.importorskip("torch")
        states, policies, values = buf.sample(2)
        assert states.shape[0] == 2
        assert policies.shape == (2, policy.size)
        assert values.shape[0] == 2


def test_replay_buffer_context_closes(tmp_path):
    state = np.zeros((12, 8, 8), dtype=np.float32)
    policy = np.zeros(get_policy_vector_size(), dtype=np.float32)
    with StreamingReplayBuffer(tmp_path) as buf:
        buf.add(state, policy, 0.1)
    with pytest.raises(AttributeError):
        buf.add(state, policy, 0.2)

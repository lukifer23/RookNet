import numpy as np
import pytest

# Skip if torch or webdataset not installed
torch = pytest.importorskip("torch")
pytest.importorskip("webdataset")

from utils.replay_buffer import StreamingReplayBuffer
from utils.move_encoder import get_policy_vector_size


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


def test_webdataset_iteration(tmp_path):
    buf = StreamingReplayBuffer(tmp_path)
    state = np.zeros((12, 8, 8), dtype=np.float32)
    policy = np.zeros(get_policy_vector_size(), dtype=np.float32)
    buf.add(state, policy, 0.5)
    buf.close()
    ds = buf.webdataset(shuffle=0)
    sample = next(iter(ds))
    board, pol, value = sample
    assert board.shape == state.shape
    assert pol.shape == policy.shape
    assert float(value) == 0.5

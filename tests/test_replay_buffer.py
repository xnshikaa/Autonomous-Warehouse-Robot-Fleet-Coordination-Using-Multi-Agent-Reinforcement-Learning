import numpy as np
import pytest
import torch

from src.marl.replay_buffer import ReplayBuffer


def test_replay_buffer_push_and_length():
    buffer = ReplayBuffer(capacity=10, num_agents=5, observation_dim=50, state_dim=250)
    assert len(buffer) == 0
    assert not buffer.can_sample(5)

    obs = np.zeros((5, 50), dtype=np.float32)
    state = np.zeros((250,), dtype=np.float32)
    actions = [0, 1, 2, 3, 0]
    reward = 1.5
    next_obs = np.ones((5, 50), dtype=np.float32)
    next_state = np.ones((250,), dtype=np.float32)
    terminated = False

    buffer.push(obs, state, actions, reward, next_obs, next_state, terminated)
    assert len(buffer) == 1
    assert buffer.can_sample(1)
    assert not buffer.can_sample(2)


def test_replay_buffer_capacity_overflow():
    buffer = ReplayBuffer(capacity=3, num_agents=5, observation_dim=50, state_dim=250)

    for i in range(5):
        obs = np.full((5, 50), i, dtype=np.float32)
        state = np.full((250,), i, dtype=np.float32)
        actions = [i % 4] * 5
        reward = float(i)
        next_obs = np.full((5, 50), i + 1, dtype=np.float32)
        next_state = np.full((250,), i + 1, dtype=np.float32)
        buffer.push(obs, state, actions, reward, next_obs, next_state, False)

    assert len(buffer) == 3
    # Check that older items 0 and 1 were overwritten
    stored_rewards = [item["rewards"] for item in buffer.storage]
    assert 0.0 not in stored_rewards
    assert 1.0 not in stored_rewards
    assert set(stored_rewards) == {2.0, 3.0, 4.0}


def test_replay_buffer_sampling_shapes_and_types():
    buffer = ReplayBuffer(capacity=10, num_agents=5, observation_dim=50, state_dim=250, seed=42)

    for i in range(8):
        obs = np.full((5, 50), i, dtype=np.float32)
        state = np.full((250,), i, dtype=np.float32)
        actions = [0, 1, 2, 3, 1]
        reward = float(i * 0.5)
        next_obs = np.full((5, 50), i + 0.1, dtype=np.float32)
        next_state = np.full((250,), i + 0.1, dtype=np.float32)
        buffer.push(obs, state, actions, reward, next_obs, next_state, i == 7)

    batch = buffer.sample(batch_size=4)

    assert batch["observations"].shape == (4, 5, 50)
    assert batch["global_states"].shape == (4, 250)
    assert batch["actions"].shape == (4, 5)
    assert batch["rewards"].shape == (4,)
    assert batch["next_observations"].shape == (4, 5, 50)
    assert batch["next_global_states"].shape == (4, 250)
    assert batch["terminated"].shape == (4,)

    assert batch["observations"].dtype == torch.float32
    assert batch["actions"].dtype == torch.long
    assert batch["rewards"].dtype == torch.float32


def test_replay_buffer_sampling_before_enough_data_raises_error():
    buffer = ReplayBuffer(capacity=10)
    buffer.push(
        np.zeros((5, 50)),
        np.zeros((250,)),
        [0, 0, 0, 0, 0],
        0.0,
        np.zeros((5, 50)),
        np.zeros((250,)),
        False,
    )
    with pytest.raises(ValueError, match="Cannot sample 4 transitions"):
        buffer.sample(4)


def test_replay_buffer_deterministic_sampling_with_seed():
    b1 = ReplayBuffer(capacity=20, seed=123)
    b2 = ReplayBuffer(capacity=20, seed=123)

    for i in range(10):
        t = (
            np.full((5, 50), i, dtype=np.float32),
            np.full((250,), i, dtype=np.float32),
            [i % 4] * 5,
            float(i),
            np.full((5, 50), i + 1, dtype=np.float32),
            np.full((250,), i + 1, dtype=np.float32),
            False,
        )
        b1.push(*t)
        b2.push(*t)

    batch1 = b1.sample(5)
    batch2 = b2.sample(5)

    assert torch.equal(batch1["rewards"], batch2["rewards"])
    assert torch.equal(batch1["actions"], batch2["actions"])


def test_replay_buffer_invalid_shapes_raise():
    buffer = ReplayBuffer(capacity=10, num_agents=5, observation_dim=50, state_dim=250)

    # Wrong obs shape
    with pytest.raises(ValueError, match="Expected observations shape"):
        buffer.push(
            np.zeros((4, 50)),
            np.zeros((250,)),
            [0] * 5,
            0.0,
            np.zeros((5, 50)),
            np.zeros((250,)),
            False,
        )

    # Wrong global_state shape
    with pytest.raises(ValueError, match="Expected global_state shape"):
        buffer.push(
            np.zeros((5, 50)),
            np.zeros((100,)),
            [0] * 5,
            0.0,
            np.zeros((5, 50)),
            np.zeros((250,)),
            False,
        )


def test_replay_buffer_serialization():
    buffer = ReplayBuffer(capacity=10)
    for i in range(3):
        buffer.push(
            np.full((5, 50), i, dtype=np.float32),
            np.full((250,), i, dtype=np.float32),
            [i % 4] * 5,
            float(i),
            np.full((5, 50), i + 1, dtype=np.float32),
            np.full((250,), i + 1, dtype=np.float32),
            False,
        )

    state_dict = buffer.state_dict()
    new_buffer = ReplayBuffer(capacity=10)
    new_buffer.load_state_dict(state_dict)

    assert len(new_buffer) == 3
    assert new_buffer.position == 3

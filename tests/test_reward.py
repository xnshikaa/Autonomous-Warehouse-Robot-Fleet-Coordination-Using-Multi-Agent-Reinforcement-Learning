import pytest

from src.ai.reward_function import RewardFunction


def test_progress_reward():
    reward_function = RewardFunction()

    reward = reward_function.calculate(
        previous_distance=10,
        current_distance=9
    )

    assert reward == pytest.approx(0.09)


def test_delivery_reward():
    reward_function = RewardFunction()

    reward = reward_function.calculate(
        previous_distance=10,
        current_distance=10,
        delivered=True
    )

    assert reward == pytest.approx(9.99)


def test_collision_penalty():
    reward_function = RewardFunction()

    reward = reward_function.calculate(
        previous_distance=10,
        current_distance=10,
        collision=True
    )

    assert reward == pytest.approx(-10.01)


def test_danger_penalty():
    reward_function = RewardFunction()

    reward = reward_function.calculate(
        previous_distance=10,
        current_distance=10,
        danger=True
    )

    assert reward == pytest.approx(-20.01)


def test_combined_reward():
    reward_function = RewardFunction()

    reward = reward_function.calculate(
        previous_distance=10,
        current_distance=9,
        delivered=True
    )

    assert reward == pytest.approx(10.09)
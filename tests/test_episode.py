import pytest

from src.ai.episode import EpisodeManager


def test_episode_starts_at_zero():
    episode = EpisodeManager(max_steps=100)

    assert episode.current_step == 0


def test_episode_not_terminated_when_tasks_remain():
    episode = EpisodeManager(max_steps=100)

    assert episode.is_terminated(tasks_remaining=5) is False


def test_episode_terminates_when_all_tasks_completed():
    episode = EpisodeManager(max_steps=100)

    assert episode.is_terminated(tasks_remaining=0) is True


def test_episode_terminates_at_max_steps():
    episode = EpisodeManager(max_steps=10)

    for _ in range(10):
        episode.step()

    assert episode.current_step == 10
    assert episode.is_terminated(tasks_remaining=5) is True


def test_episode_reset():
    episode = EpisodeManager(max_steps=10)

    episode.step()
    episode.step()

    episode.reset()

    assert episode.current_step == 0


def test_termination_reason_all_tasks_completed():
    episode = EpisodeManager(max_steps=100)

    assert episode.termination_reason(0) == "all_tasks_completed"


def test_termination_reason_max_steps():
    episode = EpisodeManager(max_steps=5)

    for _ in range(5):
        episode.step()

    assert episode.termination_reason(2) == "max_steps_reached"


def test_no_termination_reason():
    episode = EpisodeManager(max_steps=100)

    episode.step()

    assert episode.termination_reason(5) is None


def test_negative_tasks_are_invalid():
    episode = EpisodeManager(max_steps=100)

    with pytest.raises(ValueError):
        episode.is_terminated(-1)


def test_invalid_max_steps():
    with pytest.raises(ValueError):
        EpisodeManager(max_steps=0)
from src.marl.iql_comparison import IQLComparison


def test_iql_comparison_initialization():
    comparison = IQLComparison(
        episodes=2,
        max_steps=5,
        seed=42,
    )

    assert comparison.episodes == 2
    assert comparison.max_steps == 5
    assert comparison.num_robots == 5
    assert comparison.observation_dim == 50
    assert comparison.action_dim == 4


def test_iql_comparison_runs():
    comparison = IQLComparison(
        episodes=2,
        max_steps=5,
        seed=42,
    )

    learners = comparison._create_iql_learners()

    results = comparison.compare(learners)

    assert "iql" in results
    assert "rule_based" in results
    assert "difference" in results

    assert "average_reward" in results["iql"]
    assert "average_reward" in results["rule_based"]

    assert "reward_difference" in results["difference"]


def test_iql_comparison_values_are_numeric():
    comparison = IQLComparison(
        episodes=2,
        max_steps=5,
        seed=42,
    )

    learners = comparison._create_iql_learners()

    results = comparison.compare(learners)

    assert isinstance(
        results["iql"]["average_reward"],
        float,
    )

    assert isinstance(
        results["rule_based"]["average_reward"],
        float,
    )

    assert isinstance(
        results["difference"]["reward_difference"],
        float,
    )
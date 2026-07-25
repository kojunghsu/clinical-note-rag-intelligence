from src.live_ragas import _overall_level


def test_overall_high_requires_both_scores_at_least_point_75() -> None:
    assert _overall_level(0.75, 0.75) == "High"
    assert _overall_level(0.74, 0.90) != "High"


def test_overall_medium_requires_both_scores_at_least_point_50() -> None:
    assert _overall_level(0.50, 0.50) == "Medium"
    assert _overall_level(0.49, 0.90) == "Low"

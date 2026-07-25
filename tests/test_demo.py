import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_demo_matches_streamlit_ragas_panel() -> None:
    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")

    assert "faithfulness ${c.faithfulness.toFixed(4)}" in html
    assert "answer_relevancy ${c.relevancy.toFixed(4)}" in html
    assert "Overall ${c.overall}" in html
    assert "Offline RAGAS evaluation" not in html


def test_demo_contains_three_evaluated_cases_and_deidentified_evidence() -> None:
    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")

    assert "faithfulness:0.8261, relevancy:0.9640" in html
    assert "faithfulness:0.7500, relevancy:0.8981" in html
    assert "faithfulness:0.5000, relevancy:0.8176" in html
    assert "DEIDENTIFIED-01" in html
    assert "DEIDENTIFIED-09" in html


def test_public_evaluation_summary_contains_no_context_text() -> None:
    summary = json.loads(
        (ROOT / "evaluation_results" / "evaluation_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["evaluated_questions"] == 12
    assert summary["full_set_means"]["faithfulness"] == 0.4434
    assert "retrieved_contexts" not in summary

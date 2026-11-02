"""Unit tests for deliberation transcript PDF generation (src/transcripts_pdf.py)."""

import json
from pathlib import Path
from typing import List

import pytest

from src.schemas import DeliberationTranscript
from src.transcripts_pdf import (
    convert_transcript_json_to_pdf,
    convert_transcripts_to_pdf,
    format_statement_html,
)


@pytest.fixture
def sample_transcripts_data() -> List[DeliberationTranscript]:
    """Provide a minimal valid deliberation transcript structure."""
    return [
        {
            "item_id": "TEST_01",
            "item_title": "Test Public Policy Dilemma",
            "dyad_id": "D1",
            "agents": ["technocrat", "ideologue"],
            "trial_index": 0,
            "turns": [
                {
                    "turn_index": 0,
                    "speaker": "Technocratic Pragmatist (P1)",
                    "persona_id": "technocrat",
                    "statement": "We must evaluate the fiscal trade-offs with <b>rigor</b> & care.",
                },
                {
                    "turn_index": 1,
                    "speaker": "Ideological Advocate (P2)",
                    "persona_id": "ideologue",
                    "statement": "Thinking Process:\n\n1. Analyze stance.\n2. Defend community values.\n\nCommunity identity is not negotiable.",
                },
            ],
        },
        {
            "item_id": "TEST_01",
            "item_title": "Test Public Policy Dilemma",
            "dyad_id": "D2",
            "agents": ["technocrat", "populist"],
            "trial_index": 1,
            "turns": [
                {
                    "turn_index": 0,
                    "speaker": "Technocratic Pragmatist (P1)",
                    "persona_id": "technocrat",
                    "statement": "Here is a balanced compromise mechanism.",
                },
                {
                    "turn_index": 1,
                    "speaker": "Disaffected Populist (P3)",
                    "persona_id": "populist",
                    "statement": "Everyday citizens are feeling the strain <of costs>.",
                },
            ],
        },
    ]


def test_format_statement_html() -> None:
    """Verify statement HTML escaping, markdown conversion, and reasoning trace tags."""
    # Empty statement
    assert format_statement_html("") == "<i>[Empty statement]</i>"
    assert format_statement_html("   ") == "<i>[Empty statement]</i>"

    # XML characters escaping
    raw = "Policy A < Policy B & Policy C > Policy D"
    formatted = format_statement_html(raw)
    assert "&lt;" in formatted
    assert "&gt;" in formatted
    assert "&amp;" in formatted

    # Markdown bolding
    bold_raw = "This is **critical**."
    assert "<b>critical</b>" in format_statement_html(bold_raw)

    # Thinking Process trace
    tp_raw = "Thinking Process:\n\n1. Plan statement."
    formatted_tp = format_statement_html(tp_raw)
    assert "[Model Internal Reasoning Trace]" in formatted_tp
    assert "<br/>" in formatted_tp


def test_convert_transcript_json_to_pdf_single(
    tmp_path: Path, sample_transcripts_data: List[DeliberationTranscript]
) -> None:
    """Verify single transcript JSON file converts to a non-empty, valid PDF."""
    json_path = tmp_path / "item01_test_transcript.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sample_transcripts_data, f, indent=2)

    pdf_path = convert_transcript_json_to_pdf(json_path)

    assert pdf_path.exists()
    assert pdf_path.name == "item01_test_transcript.pdf"
    assert pdf_path.stat().st_size > 1000

    # Verify PDF magic bytes (%PDF)
    with open(pdf_path, "rb") as f:
        header = f.read(5)
        assert header.startswith(b"%PDF")


def test_convert_transcripts_to_pdf_batch(
    tmp_path: Path, sample_transcripts_data: List[DeliberationTranscript]
) -> None:
    """Verify batch conversion across multiple transcript files in a directory."""
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, 3):
        json_file = transcripts_dir / f"item0{i}_topic_transcript.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_transcripts_data, f, indent=2)

    generated_pdfs = convert_transcripts_to_pdf(transcripts_dir)

    assert len(generated_pdfs) == 2
    for pdf in generated_pdfs:
        assert pdf.exists()
        assert pdf.suffix == ".pdf"
        assert pdf.stat().st_size > 1000


def test_convert_transcript_long_statement(tmp_path: Path) -> None:
    """Verify that unusually long statements (>3000 chars) do not trigger ReportLab LayoutError."""
    long_statement = "This is a detailed deliberative argument. " * 150  # ~6300 chars
    data: List[DeliberationTranscript] = [
        {
            "item_id": "LONG_01",
            "item_title": "Dilemma with Long Text",
            "dyad_id": "D1",
            "agents": ["technocrat", "ideologue"],
            "trial_index": 0,
            "turns": [
                {
                    "turn_index": 0,
                    "speaker": "Technocratic Pragmatist (P1)",
                    "persona_id": "technocrat",
                    "statement": long_statement,
                }
            ],
        }
    ]
    json_path = tmp_path / "item_long_transcript.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    pdf_path = convert_transcript_json_to_pdf(json_path)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 1000


def test_convert_transcripts_empty_dir_raises(tmp_path: Path) -> None:
    """Verify non-existent directory raises FileNotFoundError."""
    missing_dir = tmp_path / "non_existent"
    with pytest.raises(FileNotFoundError):
        convert_transcripts_to_pdf(missing_dir)

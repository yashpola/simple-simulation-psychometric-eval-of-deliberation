# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingTypeStubs=false, reportUntypedBaseClass=false
"""Module for converting JSON deliberation transcripts into formatted, reviewable PDFs.

Generates human-readable, publication-grade PDF documents for each topic dilemma,
detailing multi-turn citizen dialogue across all experimental trials and dyad pairings.
"""

import json
import re
import xml.sax.saxutils as saxutils
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.schemas import DeliberationTranscript, DeliberationTurn


DYAD_METADATA: Dict[str, Dict[str, str]] = {
    "D1": {
        "name": "Technocrat vs Ideologue",
        "dynamic": "Ideological Friction",
    },
    "D2": {
        "name": "Technocrat vs Populist",
        "dynamic": "Asymmetric Vulnerability",
    },
    "D3": {
        "name": "Ideologue vs Populist",
        "dynamic": "Polarized Stalemate",
    },
    "D4": {
        "name": "Ideologue vs Everyman",
        "dynamic": "Rhetorical Dominance",
    },
    "D5": {
        "name": "Populist vs Everyman",
        "dynamic": "Grassroots Realignment",
    },
    "D6": {
        "name": "Populist vs Technocrat",
        "dynamic": "Anti-Elitist Pushback",
    },
    "D7": {
        "name": "Everyman vs Technocrat",
        "dynamic": "Pragmatic Grounding",
    },
    "D8": {
        "name": "Everyman vs Ideologue",
        "dynamic": "Moral Grounding",
    },
}

PERSONA_THEMES: Dict[str, Dict[str, Any]] = {
    "technocrat": {
        "title": "Technocratic Pragmatist (P1)",
        "badge_bg": colors.HexColor("#EFF6FF"),
        "badge_text": colors.HexColor("#1E40AF"),
        "border": colors.HexColor("#BFDBFE"),
    },
    "ideologue": {
        "title": "Ideological Advocate (P2)",
        "badge_bg": colors.HexColor("#FEF2F2"),
        "badge_text": colors.HexColor("#991B1B"),
        "border": colors.HexColor("#FECACA"),
    },
    "populist": {
        "title": "Disaffected Populist (P3)",
        "badge_bg": colors.HexColor("#FFF7ED"),
        "badge_text": colors.HexColor("#C2410C"),
        "border": colors.HexColor("#FED7AA"),
    },
    "everyman": {
        "title": "Open-Minded Everyman (P4)",
        "badge_bg": colors.HexColor("#F0FDF4"),
        "badge_text": colors.HexColor("#15803D"),
        "border": colors.HexColor("#BBF7D0"),
    },
}

DEFAULT_THEME: Dict[str, Any] = {
    "title": "Citizen Deliberator",
    "badge_bg": colors.HexColor("#F1F5F9"),
    "badge_text": colors.HexColor("#334155"),
    "border": colors.HexColor("#CBD5E1"),
}


class NumberedCanvas(canvas.Canvas):
    """Canvas that performs a two-pass calculation for total page counts and running headers/footers."""

    doc_title: str = "Deliberation Transcripts"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def _draw_header_footer(self, total_pages: int) -> None:
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 11 * inch - 28, self.doc_title[:80])
            self.drawRightString(8.5 * inch - 36, 11 * inch - 28, "Deliberative IRT Simulation Benchmark")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(36, 11 * inch - 32, 8.5 * inch - 36, 11 * inch - 32)

        # Running Footer
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(36, 36, 8.5 * inch - 36, 36)
        self.drawString(36, 24, "Confidential Citizen Deliberation Battery · Standardized Singapore Policy Dilemma")
        self.drawRightString(8.5 * inch - 36, 24, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()


def format_statement_html(statement: str) -> str:
    """Format raw statement text into safe, readable HTML for ReportLab Paragraphs."""
    if not statement or not statement.strip():
        return "<i>[Empty statement]</i>"

    # 1. Escape XML characters
    escaped = saxutils.escape(statement)

    # 2. Convert markdown bold **text** to <b>text</b>
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)

    # 3. Clean up Thinking Process headers if present to make them visually distinguishable
    if "Thinking Process:" in escaped:
        escaped = escaped.replace(
            "Thinking Process:",
            '<font color="#6B7280"><b>[Model Internal Reasoning Trace]</b></font><br/>',
        )

    # 4. Convert newlines to break tags
    html_formatted = escaped.replace("\n", "<br/>")
    return html_formatted


def convert_transcript_json_to_pdf(
    json_path: Union[str, Path],
    output_pdf_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Convert a single topic transcript JSON into a formatted PDF document.

    Args:
        json_path: Path to the *_transcript.json file.
        output_pdf_path: Optional destination path. Defaults to same directory with .pdf extension.

    Returns:
        Path of the generated PDF.
    """
    p_json = Path(json_path)
    if not p_json.exists():
        raise FileNotFoundError(f"Transcript JSON not found: {json_path}")

    p_pdf = Path(output_pdf_path) if output_pdf_path else p_json.with_suffix(".pdf")

    with open(p_json, "r", encoding="utf-8") as f:
        trials_data: List[DeliberationTranscript] = json.load(f)

    if not trials_data:
        raise ValueError(f"Transcript JSON contains no trials: {json_path}")

    first_trial = trials_data[0]
    item_id = first_trial.get("item_id", "ITEM_XX")
    item_title = first_trial.get("item_title", p_json.stem)
    total_trials = len(trials_data)
    total_turns = sum(len(t.get("turns", [])) for t in trials_data)

    doc = SimpleDocTemplate(
        str(p_pdf),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=42,
        bottomMargin=46,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )

    meta_style = ParagraphStyle(
        "CoverMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#475569"),
    )

    trial_title_style = ParagraphStyle(
        "TrialTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
    )

    trial_sub_style = ParagraphStyle(
        "TrialSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#64748B"),
    )

    speaker_title_style = ParagraphStyle(
        "SpeakerTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
    )

    turn_body_style = ParagraphStyle(
        "TurnBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1F2937"),
        spaceBefore=3,
        spaceAfter=4,
    )

    story: List[Any] = []

    # Document Header Title Block
    story.append(Paragraph(f"Deliberation Transcripts: {item_title}", title_style))
    meta_text = (
        f"<b>Item ID:</b> <font color='#2563EB'>{item_id}</font> &nbsp;|&nbsp; "
        f"<b>Sessions:</b> {total_trials} Trials (4 Dyads &times; 5 Trials) &nbsp;|&nbsp; "
        f"<b>Total Dialogue Turns:</b> {total_turns} turns &nbsp;|&nbsp; "
        f"<b>Source:</b> <font color='#64748B'>{p_json.name}</font>"
    )
    story.append(Paragraph(meta_text, meta_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=14))

    # Iterate Through Each Trial
    for trial_idx, trial in enumerate(trials_data):
        dyad_id = trial.get("dyad_id", "D1")
        dyad_info = DYAD_METADATA.get(dyad_id, {"name": "-".join(trial.get("agents", [])), "dynamic": "Dyadic Dialogue"})
        trial_num = trial.get("trial_index", trial_idx) + 1
        turns: List[DeliberationTurn] = trial.get("turns", [])

        # Trial Header Banner Table
        banner_content = [
            [
                Paragraph(
                    f"<b>Trial {trial_num} of {total_trials // 4}</b> &nbsp;|&nbsp; "
                    f"<b>Dyad {dyad_id}:</b> {dyad_info['name']}",
                    trial_title_style,
                ),
                Paragraph(
                    f"<b>Dynamic:</b> {dyad_info['dynamic']} &nbsp;|&nbsp; <b>Turns:</b> {len(turns)}",
                    trial_sub_style,
                ),
            ]
        ]
        banner_table = Table(banner_content, colWidths=[5.0 * inch, 2.5 * inch])
        banner_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
                    ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E1")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )

        trial_flowables: List[Any] = [banner_table, Spacer(1, 6)]

        for turn in turns:
            turn_idx = turn.get("turn_index", 0) + 1
            speaker_name = turn.get("speaker", "Speaker")
            persona_id = turn.get("persona_id", "citizen")
            theme = PERSONA_THEMES.get(persona_id, DEFAULT_THEME)

            speaker_html = (
                f"<font color='{theme['badge_text'].hexval()}'><b>Turn {turn_idx} of {len(turns)}</b>"
                f" &nbsp;&bull;&nbsp; <b>{speaker_name}</b></font>"
            )

            statement_raw = turn.get("statement", "")
            statement_html = format_statement_html(statement_raw)

            # Avoid Table cell overflow for unusually long statements
            if len(statement_raw) <= 1500:
                turn_content = [
                    [Paragraph(speaker_html, speaker_title_style)],
                    [Paragraph(statement_html, turn_body_style)],
                ]
                turn_table = Table(turn_content, colWidths=[7.5 * inch])
                turn_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), theme["badge_bg"]),
                            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#FFFFFF")),
                            ("BOX", (0, 0), (-1, -1), 0.5, theme["border"]),
                            ("LINEBELOW", (0, 0), (-1, 0), 0.5, theme["border"]),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                            ("LEFTPADDING", (0, 0), (-1, -1), 8),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ]
                    )
                )
                trial_flowables.append(KeepTogether([turn_table, Spacer(1, 4)]))
            else:
                header_table = Table([[Paragraph(speaker_html, speaker_title_style)]], colWidths=[7.5 * inch])
                header_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), theme["badge_bg"]),
                            ("BOX", (0, 0), (-1, -1), 0.5, theme["border"]),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                            ("LEFTPADDING", (0, 0), (-1, -1), 8),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ]
                    )
                )
                trial_flowables.append(header_table)
                trial_flowables.append(Spacer(1, 2))
                paras = statement_html.split("<br/><br/>")
                for para in paras:
                    if para.strip():
                        trial_flowables.append(Paragraph(para.strip(), turn_body_style))
                        trial_flowables.append(Spacer(1, 3))
                trial_flowables.append(Spacer(1, 4))

        trial_flowables.append(Spacer(1, 8))
        story.extend(trial_flowables)

    # Build Document with Item-specific NumberedCanvas subclass
    class ItemNumberedCanvas(NumberedCanvas):
        doc_title = f"{item_id}: {item_title}"

    doc.build(story, canvasmaker=ItemNumberedCanvas)
    return p_pdf


def convert_transcripts_to_pdf(transcripts_dir: Union[str, Path]) -> List[Path]:
    """Batch convert all *_transcript.json files in a directory to PDF.

    Args:
        transcripts_dir: Directory containing transcript JSON files.

    Returns:
        List of generated PDF file paths.
    """
    t_dir = Path(transcripts_dir)
    if not t_dir.exists():
        raise FileNotFoundError(f"Transcripts directory does not exist: {transcripts_dir}")

    json_files = sorted(t_dir.glob("*_transcript.json"))
    if not json_files:
        # Fallback to any .json file in transcripts folder
        json_files = sorted(t_dir.glob("*.json"))

    generated_pdfs: List[Path] = []
    for json_file in json_files:
        pdf_path = convert_transcript_json_to_pdf(json_file)
        generated_pdfs.append(pdf_path)

    return generated_pdfs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert deliberation transcripts from JSON to PDF")
    parser.add_argument(
        "--dir",
        type=str,
        default="data/exp001/transcripts",
        help="Directory containing transcript JSON files",
    )
    args = parser.parse_args()

    pdfs = convert_transcripts_to_pdf(args.dir)
    print(f"Successfully generated {len(pdfs)} PDF transcripts in {args.dir}:")
    for pdf in pdfs:
        print(f"  • {pdf.name}")

"""Generates comprehensive Markdown report synthesizing CTT, IRT, and discourse findings.

Adheres to:
- Guideline 1: Built-in typing and TypedDict inputs.
- Guideline 2: Functional separation for report synthesis.
- Simplified benchmark observing latent deliberative competence across diverse citizen groups and policy items.
"""

from typing import Any, Dict, Optional
from datetime import datetime
from src.schemas import (
    MasterConfig,
    PolicyItem,
    PsychometricResults,
    IRTMetrics,
    IRTDeliberators,
    IRTResults,
    CTTMetrics,
    ExecutionMetadata,
)
from src.personas import get_persona_metadata


def generate_markdown_report(
    config: MasterConfig,
    metrics: PsychometricResults,
    items_data: Dict[str, PolicyItem],
    output_path: str,
    execution_metadata: Optional[ExecutionMetadata] = None
) -> str:
    """Compile final_report.md synthesizing deliberative psychometrics and governance insights."""
    ctt_data: Dict[str, CTTMetrics] = metrics.get("ctt", {})
    irt_block: Optional[IRTResults] = metrics.get("irt")
    irt_items: Dict[str, IRTMetrics] = irt_block.get("items", {}) if irt_block else {}
    deliberators: Optional[IRTDeliberators] = irt_block.get("deliberators") if irt_block else None

    dyad_thetas: Dict[str, float] = deliberators.get("dyad_theta", {}) if deliberators else {}
    ind_thetas: Dict[str, float] = deliberators.get("individual_theta", {}) if deliberators else {}
    persona_by_dyad: Dict[str, Dict[str, float]] = deliberators.get("persona_by_dyad_theta", {}) if deliberators else {}
    ind_success: Dict[str, float] = metrics.get("individual_success_rates", {})
    insights: Dict[str, Any] = metrics.get("insights", {})

    model_cfg = config.get("model_config", {})
    delib_cfg = config.get("deliberation_config", {})
    meta = config.get("experiment_metadata", {})
    dyads_cfg = delib_cfg.get("dyads", [])
    trials_per_item = delib_cfg.get("trials_per_item", 5)
    n_dyads = max(1, len(dyads_cfg))
    n_items = max(1, len(items_data))
    sessions_per_item = trials_per_item * n_dyads
    votes_per_item = sessions_per_item * 2
    total_sessions = trials_per_item * n_items * n_dyads
    total_votes = total_sessions * 2

    discarded_cnt = int(insights.get("discarded_ballot_count", 0))
    valid_cnt = int(insights.get("valid_votes_count", insights.get("valid_ballot_count", total_votes - discarded_cnt)))

    report_lines = [
        "# Deliberative Item Response Theory (IRT) Benchmark Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Benchmark:** {meta.get('name', 'Psychometric Deliberative Capacity Benchmark')}  ",
        f"**Model:** `{model_cfg.get('agent_model', 'openai/gpt-oss-120b')}`  ",
        f"**Execution Matrix:** {trials_per_item} trials/dyad across {n_dyads} dyads and {n_items} items ({total_sessions} total sessions, {total_votes} secret ballots)  ",
        f"**Per-Item Sample:** {sessions_per_item} deliberative sessions ({votes_per_item} individual secret ballots cast)  ",
    ]

    runtime_fmt = execution_metadata.get("runtime_formatted") if execution_metadata else None
    runtime_sec = execution_metadata.get("runtime_seconds") if execution_metadata else None
    if runtime_fmt:
        report_lines.append(f"**Runtime:** {runtime_fmt} ({runtime_sec}s)  ")
    elif runtime_sec is not None:
        report_lines.append(f"**Runtime:** {runtime_sec}s  ")

    cost_usd = execution_metadata.get("cost_usd") if execution_metadata else None
    if cost_usd is not None:
        report_lines.append(f"**Estimated Cost:** ${cost_usd:.4f} USD  ")

    report_lines.extend([
        "",
        "---",
        "",
        "## 1. Executive Summary & Core Insights",
        "",
    ])

    if discarded_cnt > 0:
        report_lines.extend([
            "> [!WARNING]",
            f"> **Data Quality Alert**: {discarded_cnt} of {total_votes} secret ballot(s) produced empty or unparseable responses and were discarded from psychometric evaluation. ({valid_cnt} valid ballots evaluated).",
            "",
        ])

    report_lines.extend([
        "This psychometric benchmark evaluates **latent deliberative capacity ($\\theta$)** across diverse citizen demographic dyads "
        "and measures **item difficulty ($b$)**, **item discrimination ($a$)**, and **distractor efficiency (DCR)** across standardized Singapore public policy dilemmas.",
        "",
        "### Key Empirical Findings:",
        f"- **Most Challenging Policy Dilemma**: Policy item `{insights.get('hardest_item', 'N/A')}` exhibited the highest intrinsic difficulty parameter ($b$).",
        f"- **Most Accessible Compromise**: Policy item `{insights.get('easiest_item', 'N/A')}` had the lowest difficulty parameter, proving most receptive to common ground.",
        f"- **Most Diagnostic Dilemma**: Policy item `{insights.get('most_discriminating_item', 'N/A')}` showed the highest discrimination parameter ($a$), reliably distinguishing high-deliberation dyads from vulnerable ones.",
        f"- **Most Vulnerable Citizen Dyad**: Dyad `{insights.get('lowest_competence_dyad', 'N/A')}` demonstrated the lowest baseline latent deliberative competence ($\\theta$).",
        f"- **Most Tempting Distractor Trap**: Option `{insights.get('most_tempting_distractor', 'N/A')}` was the most frequently chosen bad-faith heuristic across participants.",
        "",
        "---",
        "",
        "## 2. Deliberation & Psychometric Architecture",
        "",
        "```mermaid",
        "flowchart TD",
        "    subgraph Phase1 [\"1. Inputs & Configuration\"]",
        "        CFG[\"Master Config (config.json)\"]",
        "        DYADS[\"Citizen Personas & Dyads (D1 to D4)\"]",
        "        ITEMS[\"Standardized Policy Battery (Items 1 to 4)\"]",
        "        CFG --> DYADS & ITEMS",
        "    end",
        "",
        "    subgraph Phase2 [\"2. Deliberation Simulation\"]",
        "        DIALOGUE[\"Multi-Turn Dyadic Deliberation between Citizen Personas\"]",
        "    end",
        "",
        "    subgraph Phase3 [\"3. Secret Ballot Administration\"]",
        "        BALLOT[\"Randomized Secret Ballot\"]",
        "        KEY[\"Integrative Compromise\"]",
        "        TRAPS[\"Cognitive Traps (Inertia, Zero-Sum, Scapegoating)\"]",
        "        BALLOT --> KEY & TRAPS",
        "    end",
        "",
        "    subgraph Phase4 [\"4. Psychometric Modeling\"]",
        "        CTT[\"Classical Test Theory (Difficulty p, Discrimination D, DCR)\"]",
        "        IRT[\"Item Response Theory (2PL: Ability theta, Difficulty b, Discrimination a)\"]",
        "    end",
        "",
        "    subgraph Phase5 [\"5. Reorganized Experiment Artifacts (data/expXXX)\"]",
        "        INPUTS[\"inputs/ (config.json, items/)\"]",
        "        OUTPUTS[\"model_outputs/ (transcripts/, ballots.json)\"]",
        "        EVALS[\"evals/ (psychometric_evals.json, final_report.md)\"]",
        "        META[\"metadata/ (metadata.json)\"]",
        "    end",
        "",
        "    Phase1 --> Phase2",
        "    Phase2 --> Phase3",
        "    Phase3 --> Phase4",
        "    Phase4 --> Phase5",
        "```",
        "",
        "---",
        "",
        "## 3. Standardized Policy Battery Psychometrics",
        "",
        "### 3.1 Classical Test Theory (CTT) Empirical Metrics",
        "Direct observational statistics measuring empirical consensus difficulty ($p_i$), Kelley's upper-lower discrimination index ($D_i$), and cognitive distractor ratios ($\\text{DCR}$):",
        "",
        "| Item ID | Policy Dilemma Topic | Consensus Rate ($p_i$) | Discrimination ($D_i$) | DCR (B: Inertia) | DCR (C: Zero-Sum) | DCR (D: Scapegoat) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"
    ])

    for item_id, item_info in items_data.items():
        title = item_info.get("title", item_id)
        short_title = title[:34] + ("..." if len(title) > 34 else "")
        c_info = ctt_data.get(item_id)

        p_val: object = c_info["difficulty_p"] if c_info else "N/A"
        d_val: object = c_info["discrimination_index"] if c_info else "N/A"
        dcr_dict: Dict[str, float] = c_info["distractor_ratios"] if c_info else {}

        report_lines.append(
            f"| `{item_id}` | {short_title} | {p_val} | {d_val} | {dcr_dict.get('B', 0.0)} | {dcr_dict.get('C', 0.0)} | {dcr_dict.get('D', 0.0)} |"
        )

    report_lines.extend([
        "",
        "> [!NOTE]",
        "> **CTT Metric Interpretation Guide**:",
        f"> - **Consensus Rate ($p_i$)**: The proportion of deliberative sessions where *both* participants in the dyad simultaneously voted for the integrative compromise across all **{sessions_per_item} sessions** ($5\\text{{ trials}} \\times {n_dyads}\\text{{ dyads}}$).  ",
        "> - **Item Discrimination ($D_i$)**: Kelley's optimal upper/lower 27% validation split ($P_{\\text{top}} - P_{\\text{bottom}}$), measuring separation between top- and bottom-performing dyads.  ",
        f"> - **DCR (Distractor-to-Correct Ratio)**: A relative ratio against the integrative compromise, **not a percentage of total votes**: $\\text{{DCR}}_k = \\frac{{\\text{{Count}}(\\text{{Distractor }} k)}}{{\\text{{Count}}(\\text{{Integrative Compromise}})}}$. Evaluated across all **{votes_per_item} individual secret ballots** cast for that dilemma.  ",
        ">   - $\\text{DCR} > 1.0$: Dominant trap (bad-faith heuristic was chosen *more frequently* than the compromise).  ",
        ">   - $\\text{DCR} < 1.0$: Subdominant trap (the integrative compromise was chosen more frequently than this heuristic).  ",
        ">   - $\\text{DCR} = 1.0$: Parity trap (equal frequency to compromise).  ",
        ">   - *Note on vote percentages*: DCR values do not sum to 100%. The actual percentage of votes for the integrative compromise is $\\frac{1}{1 + \\text{DCR}_B + \\text{DCR}_C + \\text{DCR}_D} \\times 100\\%$.  ",
        "",
        "### 3.2 Item Response Theory (2PL IRT) Calibrated Parameters",
        "Latent difficulty threshold ($b_i$) and discrimination slope ($a_i$) calibrated via regularized Maximum Likelihood Estimation:",
        "",
        "| Item ID | Policy Dilemma Topic | Difficulty ($b_i$) | Discrimination ($a_i$) | Diagnostic Characteristic |",
        "| :--- | :--- | :---: | :---: | :--- |"
    ])

    for item_id, item_info in items_data.items():
        title = item_info.get("title", item_id)
        short_title = title[:34] + ("..." if len(title) > 34 else "")
        i_info = irt_items.get(item_id)

        b_val = i_info["difficulty_b"] if i_info else 0.0
        a_val = i_info["discrimination_a"] if i_info else 1.0

        diff_desc = "High Difficulty" if b_val > 1.0 else ("Low Difficulty" if b_val < -1.0 else "Moderate Difficulty")
        disc_desc = (
            "Very High Discrimination" if a_val >= 1.70
            else ("High Discrimination" if a_val >= 1.35
            else ("Moderate Discrimination" if a_val >= 0.65
            else "Low Discrimination"))
        )

        report_lines.append(
            f"| `{item_id}` | {short_title} | {b_val} | {a_val} | {diff_desc}, {disc_desc} |"
        )

    report_lines.extend([
        "",
        "> [!NOTE]",
        "> **IRT Metric Interpretation Guide**:",
        "> - **Item Difficulty ($b_i$)**: The latent ability threshold $\\theta$ required for a 50% probability of consensus on the integrative compromise. Higher $b_i$ indicates greater intrinsic resistance to resolution.",
        "> - **Item Discrimination ($a_i$)**: The slope of the Item Characteristic Curve (ICC) at $\\theta = b_i$. Values $a \\ge 1.70$ represent very high diagnostic separation between high- and low-competence dyads.",
        "",
        "---",
        "",
        "## 4. Latent Deliberative Ability Breakdown ($\\theta$)",
        "",
        "### 4.1 Dyadic Deliberative Competence (Consensus Requirement)",
        "Estimated collective latent competence $\\theta$ across citizen dyad configurations (an item is passed only when both participants reach consensus on the integrative compromise):",
        "",
        "| Dyad ID | Citizen Composition | Deliberative Dynamic | Collective Competence ($\\theta$) | Relative Ranking |",
        "| :--- | :--- | :--- | :---: | :---: |"
    ])

    dyad_metadata = {
        d.get("id"): (d.get("name", d.get("id")), d.get("dynamic", "Deliberative Dynamic"))
        for d in delib_cfg.get("dyads", [])
    }

    # Sort dyads by theta descending
    sorted_dyads = sorted(dyad_thetas.keys(), key=lambda k: dyad_thetas[k], reverse=True)
    rank = 1
    for d_id in sorted_dyads:
        th_val = dyad_thetas.get(d_id, 0.0)
        comp, dyn = dyad_metadata.get(d_id, ("Citizen Dyad", "Deliberative Group"))
        report_lines.append(f"| `{d_id}` | {comp} | {dyn} | {th_val} | #{rank} |")
        rank += 1

    # 4.2 Individual Persona Deliberative Competence & Success Rates
    if ind_thetas:
        report_lines.extend([
            "",
            "### 4.2 Individual Persona Deliberative Competence & Success Rates",
            "Performance metrics for each citizen persona across independent secret ballots:",
            "- **Individual Success Rate**: Empirical proportion of secret ballots where the persona voted for the integrative compromise.",
            "- **Latent Competence ($\\theta$)**: Standardized 2PL IRT ability score isolating individual deliberative capacity.",
            "",
            "| Persona ID | Role Title | Individual Success Rate | Latent Competence ($\\theta$) | Relative Ranking |",
            "| :--- | :--- | :---: | :---: | :---: |"
        ])
        sorted_personas = sorted(ind_thetas.keys(), key=lambda k: ind_thetas[k], reverse=True)
        p_rank = 1
        for p_id in sorted_personas:
            p_th = ind_thetas.get(p_id, 0.0)
            p_rate = f"{ind_success.get(p_id, 0.0) * 100:.1f}%"
            p_meta = get_persona_metadata(p_id)
            p_name = p_meta.get("name", p_id)
            report_lines.append(f"| `{p_id}` | {p_name} | {p_rate} | {p_th} | #{p_rank} |")
            p_rank += 1

    # 4.3 Partner Interaction Effects (Dyadic Conditioning)
    if persona_by_dyad:
        report_lines.extend([
            "",
            "### 4.3 Partner Interaction Effects (Dyadic Conditioning)",
            "How each individual persona's latent competence shifts depending on conversational partner in the dyad:",
            "",
            "| Persona | Partner Dynamic in Dyad | Latent Competence ($\\theta$) | Observed Shift / Context |",
            "| :--- | :--- | :---: | :--- |"
        ])
        for p_id, d_map in sorted(persona_by_dyad.items()):
            p_meta = get_persona_metadata(p_id)
            p_name = p_meta.get("name", p_id)
            for d_id, th in sorted(d_map.items()):
                comp, dyn = dyad_metadata.get(d_id, (d_id, "Dyad"))
                report_lines.append(f"| {p_name} | `{d_id}` ({dyn}) | {th} | Paired in {comp} |")

    report_lines.extend([
        "",
        "---",
        "",
        "## 5. Cognitive Trap & Distractor Vulnerability Analysis",
        "",
        "The standardized assessment tests three distinct heuristic failure modes:",
        "1. **Option B (Status Quo Inertia Trap)**: Rigid refusal to adapt; demanding infinite subsidies without addressing structural change.",
        "2. **Option C (Technocratic Zero-Sum Trap)**: Aggressive efficiency-only policy that completely disregards human/community disruption.",
        "3. **Option D (External Scapegoating / Deflection Trap)**: Deflecting structural challenges onto foreign or elite scapegoats without a viable solution.",
        "",
        "### Average Distractor Efficiency Across All Dilemmas:",
    ])

    avg_dcr = insights.get("average_dcr_by_distractor", {})
    hardest_key = str(insights.get("hardest_item", ""))
    hardest_b: object = irt_items[hardest_key]["difficulty_b"] if hardest_key in irt_items else "N/A"

    report_lines.extend([
        f"- **Option B (Status Quo Inertia)**: Average DCR = `{avg_dcr.get('B', 0.0)}`",
        f"- **Option C (Technocratic Zero-Sum)**: Average DCR = `{avg_dcr.get('C', 0.0)}`",
        f"- **Option D (External Scapegoating)**: Average DCR = `{avg_dcr.get('D', 0.0)}`",
        "",
        "---",
        "",
        "## 6. Policy Governance Implications",
        "",
        f"1. **Prioritize Facilitation on Item `{insights.get('hardest_item', 'N/A')}`**: This dilemma exhibited the greatest resistance to compromise ($b = {hardest_b}$). Policymakers and deliberative moderators should provide extra contextual briefings and transition guarantees on this topic.",
        f"2. **Mitigate Vulnerability in Dyad `{insights.get('lowest_competence_dyad', 'N/A')}`**: This citizen grouping struggled most with constructive dialogue ($\\theta = {dyad_thetas.get(insights.get('lowest_competence_dyad', ''), 'N/A')}$). Structured perspective-taking exercises and active mediation should be targeted at this demographic pairing.",
        f"3. **Inoculate Against `{insights.get('most_tempting_distractor', 'N/A')}` Traps**: The highest distractor selection was observed for Option {insights.get('most_tempting_distractor', 'N/A')}, indicating that public discourse is most susceptible to this specific bad-faith heuristic.",
        "",
        "---",
        "*Report automatically generated by `deliberative-irt-simulation` psychometrics pipeline.*"
    ])

    report_content = "\n".join(report_lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_content

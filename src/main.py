"""Master orchestrator and CLI entry point for Deliberative IRT Simulation.

Adheres to:
- Guideline 1: Strict typing schemas using TypedDict.
- Guideline 2: Separation of concerns (main.py is reserved strictly as the orchestrator and CLI).
- Guideline 3: Explicit orchestration naming (execute_simulation_run) and single-responsibility steps.
- Guideline 4: Zero hardcoded prompt strings.
- Guideline 6: Auto-incremented expXXX output directories.
"""

import argparse
import glob
import json
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    from dotenv import load_dotenv  # pyright: ignore[reportUnknownVariableType]
    _ = load_dotenv()
except ImportError:
    pass

from src.schemas import (
    MasterConfig,
    PolicyItem,
    DeliberationTranscript,
    DyadBallotRecord,
    PsychometricResults,
    ExecutionMetadata,
)
from src.llm import get_llm_client
from src.deliberation import conduct_deliberation
from src.ballot import administer_dyad_ballot
from src.psychometrics import analyze_experiment_psychometrics
from src.report import generate_markdown_report
from src.transcripts_pdf import convert_transcripts_to_pdf


def get_next_experiment_dir(base_dir: str = "data") -> Path:
    """Detect existing expXXX directories and create the next auto-incremented index."""
    os.makedirs(base_dir, exist_ok=True)
    existing_dirs = glob.glob(os.path.join(base_dir, "exp[0-9][0-9][0-9]"))
    indices: List[int] = []
    for d in existing_dirs:
        m = re.search(r"exp(\d{3,})$", d)
        if m:
            indices.append(int(m.group(1)))

    next_idx = max(indices) + 1 if indices else 0
    exp_dir = Path(base_dir) / f"exp{next_idx:03d}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def load_master_configuration(config_path: str) -> MasterConfig:
    """Load master configuration from JSON file."""
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(p, "r", encoding="utf-8") as f:
        data: MasterConfig = json.load(f)
    return data


def load_policy_battery_items(item_paths: List[str]) -> Dict[str, PolicyItem]:
    """Load standardized test battery items from individual JSON files."""
    items: Dict[str, PolicyItem] = {}
    for path in item_paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Policy battery item not found: {path}")
        with open(p, "r", encoding="utf-8") as f:
            item_data: PolicyItem = json.load(f)
        items[item_data["item_id"]] = item_data
    return items


def _save_base_artifacts(
    exp_dir: Path,
    config: MasterConfig,
    items: Dict[str, PolicyItem],
    transcripts: List[DeliberationTranscript],
    all_ballots: List[DyadBallotRecord],
) -> None:
    """Save base simulation artifacts into organized subdirectories:
    - inputs/: config.json, items/
    - model_outputs/: transcripts/, ballots.json
    """
    # (a) Inputs Subdirectory
    inputs_dir = exp_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    with open(inputs_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    items_out_dir = inputs_dir / "items"
    items_out_dir.mkdir(parents=True, exist_ok=True)
    for item_path in config.get("items", []):
        src_p = Path(item_path)
        if src_p.exists():
            shutil.copy2(src_p, items_out_dir / src_p.name)
        else:
            for item_id, item_obj in items.items():
                if item_id in src_p.name or item_obj.get("item_id") in src_p.name:
                    with open(items_out_dir / src_p.name, "w", encoding="utf-8") as f:
                        json.dump(item_obj, f, indent=2)

    # (b) Model Outputs Subdirectory
    model_outputs_dir = exp_dir / "model_outputs"
    model_outputs_dir.mkdir(parents=True, exist_ok=True)

    transcripts_out_dir = model_outputs_dir / "transcripts"
    transcripts_out_dir.mkdir(parents=True, exist_ok=True)

    item_filename_map: Dict[str, str] = {}
    for item_path in config.get("items", []):
        src_p = Path(item_path)
        if src_p.exists():
            try:
                with open(src_p, "r", encoding="utf-8") as f:
                    item_json = json.load(f)
                    if "item_id" in item_json:
                        base_stem = src_p.stem.removesuffix("_description")
                        item_filename_map[item_json["item_id"]] = base_stem
            except Exception:
                pass

    for item_id in items.keys():
        base_stem = item_filename_map.get(item_id, item_id.lower())
        fname = f"{base_stem}_transcript.json"
        item_transcripts = [t for t in transcripts if t["item_id"] == item_id]
        with open(transcripts_out_dir / fname, "w", encoding="utf-8") as f:
            json.dump(item_transcripts, f, indent=2)

    # Secret Ballots
    with open(model_outputs_dir / "ballots.json", "w", encoding="utf-8") as f:
        json.dump(all_ballots, f, indent=2)

    # Convert Deliberation Transcripts to Formatted Reviewable PDFs
    try:
        pdf_paths = convert_transcripts_to_pdf(transcripts_out_dir)
        print(f"📄 Generated {len(pdf_paths)} reviewable PDF transcripts in: {transcripts_out_dir}")
    except Exception as e:
        print(f"⚠️ Warning: Could not generate PDF transcripts: {e}")


def execute_simulation_run(
    config_path: str = "data/config.json",
    output_dir: Optional[Union[str, Path]] = None
) -> Path:
    """Primary simulation orchestrator executing the deliberative benchmark.

    All experimental parameters are defined within the configuration JSON file.
    Adheres to:
    - Guideline 3: Explicit orchestration naming and single-responsibility step dispatch.
    - Guideline 4: Total configuration abstraction.
    - Guideline 6: Dynamic experiment output indexing and organized subdirectories (inputs/, model_outputs/, evals/).
    """
    start_time = time.time()

    # 1. Load configuration
    config = load_master_configuration(config_path)

    model_cfg = config.get("model_config", {})

    # 2. Output directory resolution (Guideline 6)
    target_output_dir = output_dir or config.get("output_dir")
    if target_output_dir:
        exp_dir = Path(target_output_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
    else:
        exp_dir = get_next_experiment_dir()

    delib_cfg = config.get("deliberation_config", {})
    trials_per_item = delib_cfg.get("trials_per_item", 5)
    dyads = delib_cfg.get("dyads", [])
    turns_per_item = delib_cfg.get("turns_per_item", 4)
    mcq_option_order_seed = delib_cfg.get("mcq_option_order_seed", delib_cfg.get("seed", 42))

    # 3. Load standardized test battery
    items = load_policy_battery_items(config.get("items", []))

    # 4. Initialize LLM inference client
    client = get_llm_client(model_cfg, is_judge=False)

    print(f"🚀 Initializing Deliberative IRT Simulation Benchmark")
    print(f"📁 Output Directory: {exp_dir}")
    print(f"🤖 LLM Model: {client.model}")
    print(f"📊 Policy Items: {list(items.keys())}")
    print(f"👥 Citizen Dyads: {[d.get('id') for d in dyads]}")
    print(f"🔄 Trials per item: {trials_per_item}")

    transcripts: List[DeliberationTranscript] = []
    all_ballots: List[DyadBallotRecord] = []

    # 5. Experimental Execution Matrix
    total_steps = trials_per_item * len(items) * len(dyads)
    current_step = 0

    for trial_idx in range(trials_per_item):
        for item_id, item in items.items():
            for dyad in dyads:
                current_step += 1
                dyad_label = f"{dyad.get('id')} ({'-'.join(dyad.get('agents', []))})"
                print(f"[{current_step}/{total_steps}] Trial {trial_idx + 1} | Dyad: {dyad_label} | Item: {item_id}")

                # Step A: Conduct Dyadic Deliberation
                transcript = conduct_deliberation(
                    client=client,
                    config=config,
                    item=item,
                    dyad_id=dyad.get("id", "D1"),
                    agents=dyad.get("agents", []),
                    trial_index=trial_idx,
                    turns_per_item=turns_per_item
                )
                transcripts.append(transcript)

                # Step B: Administer Independent Secret Ballot
                summary_lines = [f"{t['speaker']}: {t['statement']}" for t in transcript["turns"]]
                transcript_summary = "\n".join(summary_lines)

                ballot_record = administer_dyad_ballot(
                    client=client,
                    config=config,
                    agents=dyad.get("agents", []),
                    item=item,
                    dyad_id=dyad.get("id", "D1"),
                    trial_index=trial_idx,
                    transcript_summary=transcript_summary,
                    mcq_option_order_seed=(mcq_option_order_seed + current_step * 17)
                )
                all_ballots.append(ballot_record)

    # 6. Save Base Simulation Artifacts (inputs/ and model_outputs/)
    _save_base_artifacts(exp_dir, config, items, transcripts, all_ballots)

    # 7. Quality Assurance Validation: Non-Silent Failure Detection
    total_statements = sum(len(t.get("turns", [])) for t in transcripts)
    empty_statements = sum(
        1 for t in transcripts for turn in t.get("turns", []) if not turn.get("statement", "").strip()
    )
    all_deliberations_empty = bool(total_statements > 0 and empty_statements == total_statements)

    total_ballots = sum(len(b.get("votes", {})) for b in all_ballots)
    empty_ballots = sum(
        1
        for b in all_ballots
        for v in b.get("votes", {}).values()
        if not v.get("is_valid", False) or not v.get("raw_response", "").strip()
    )
    all_ballots_empty = bool(total_ballots > 0 and empty_ballots == total_ballots)

    if all_deliberations_empty or all_ballots_empty:
        causes: List[str] = []
        if all_deliberations_empty:
            causes.append("All deliberation turn statements were empty")
        if all_ballots_empty:
            causes.append("All secret ballots produced empty or unparseable responses")
        failure_summary = " and ".join(causes)

        end_time = time.time()
        runtime_seconds = round(end_time - start_time, 2)
        minutes = int(runtime_seconds // 60)
        seconds = runtime_seconds % 60
        runtime_formatted = f"{minutes}m {seconds:.1f}s" if minutes > 0 else f"{seconds:.2f}s"

        cost_usd = 0.0
        if hasattr(client, "get_cost_usd"):
            cost_usd = client.get_cost_usd()

        metadata_dir = exp_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        aborted_metadata: ExecutionMetadata = {
            "experiment_name": config.get("experiment_metadata", {}).get("name", "Aborted Run"),
            "model": model_cfg.get("agent_model", getattr(client, "model", "unknown")),
            "cost_usd": cost_usd,
            "runtime_seconds": runtime_seconds,
            "runtime_formatted": runtime_formatted,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "timestamp": datetime.now().isoformat(),
        }
        with open(metadata_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(aborted_metadata, f, indent=2)

        border = "=" * 80
        print(f"\n{border}")
        print("🚨 CRITICAL ERROR: RUN FAILURE DETECTED — ABORTING FINAL REPORT 🚨")
        print(border)
        print(f"Cause: {failure_summary}.")
        print("Model generation tokens may have been exhausted by model reasoning,")
        print("or the endpoint returned empty completions across all agents.")
        print("\nDiagnostics:")
        print(f"  • Deliberation Statements: {empty_statements}/{total_statements} empty")
        print(f"  • Secret Ballots: {empty_ballots}/{total_ballots} empty/invalid")
        print(f"\nBase diagnostics (config, items, transcripts, ballots)")
        print(f"have been saved to: {exp_dir / 'inputs'} and {exp_dir / 'model_outputs'}")
        print(f"Execution metadata saved to: {metadata_dir / 'metadata.json'}")
        print("Psychometric evaluation (evals/) has been ABORTED to avoid presenting contaminated data.")
        print(f"{border}\n")
        return exp_dir

    # 8. Psychometrics Analysis (CTT & IRT)
    print("📈 Fitting CTT and 2PL IRT Psychometric Models...")
    psychometric_results: PsychometricResults = analyze_experiment_psychometrics(
        ballots=all_ballots,
        item_ids=list(items.keys())
    )

    # 9. Compute Runtime and Cost Metadata (Saved to metadata/metadata.json)
    end_time = time.time()
    runtime_seconds = round(end_time - start_time, 2)
    minutes = int(runtime_seconds // 60)
    seconds = runtime_seconds % 60
    runtime_formatted = f"{minutes}m {seconds:.1f}s" if minutes > 0 else f"{seconds:.2f}s"

    cost_usd = 0.0
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if hasattr(client, "get_cost_usd"):
        cost_usd = client.get_cost_usd()
    if hasattr(client, "get_token_usage"):
        token_usage = client.get_token_usage()

    metadata_dir = exp_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata_payload: ExecutionMetadata = {
        "experiment_name": config.get("experiment_metadata", {}).get("name", "Psychometric Deliberative Capacity Benchmark"),
        "model": model_cfg.get("agent_model", getattr(client, "model", "unknown")),
        "cost_usd": cost_usd,
        "runtime_seconds": runtime_seconds,
        "runtime_formatted": runtime_formatted,
        "prompt_tokens": token_usage.get("prompt_tokens", 0),
        "completion_tokens": token_usage.get("completion_tokens", 0),
        "total_tokens": token_usage.get("total_tokens", 0),
        "timestamp": datetime.now().isoformat(),
    }
    with open(metadata_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    # 10. Persist Evaluative Artifacts (evals/)
    evals_dir = exp_dir / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)

    print("💾 Saving psychometric evaluation results and final synthesis report...")
    with open(evals_dir / "psychometric_evals.json", "w", encoding="utf-8") as f:
        json.dump(psychometric_results, f, indent=2)

    report_path = evals_dir / "final_report.md"
    generate_markdown_report(
        config=config,
        metrics=psychometric_results,
        items_data=items,
        output_path=str(report_path),
        execution_metadata=metadata_payload
    )

    print(f"\n🎉 Simulation Run Completed Successfully!")
    print(f"⏱️ Runtime: {runtime_formatted} ({runtime_seconds}s)")
    print(f"💰 Estimated Cost: ${cost_usd:.4f} USD")
    print(f"📄 Report written to: {report_path}")
    return exp_dir


def main() -> None:
    """CLI argument parsing entry point.

    Accepts ONLY the path to the configuration JSON file.
    All experimental parameters are defined within the configuration JSON.
    """
    parser = argparse.ArgumentParser(
        description="Deliberative IRT Simulation: Evaluating Persona Competence & Policy Items via Item Response Theory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data/config.json",
        help="Path to master config JSON (default: data/config.json)"
    )

    args = parser.parse_args()

    execute_simulation_run(config_path=args.config)


if __name__ == "__main__":
    main()

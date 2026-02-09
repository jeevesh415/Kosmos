#!/usr/bin/env python3
"""
Persona-based evaluation orchestrator for Kosmos.

Loads a persona YAML definition, creates a versioned run directory,
and executes Tier 1 (automated evaluation) by invoking scientific_evaluation.py
with persona-specific parameters.

Usage:
    # Run Tier 1 for a persona
    python evaluation/personas/run_persona_eval.py \
        --persona 001_enzyme_kinetics_biologist --tier 1

    # Dry run (show what would execute)
    python evaluation/personas/run_persona_eval.py \
        --persona 001_enzyme_kinetics_biologist --tier 1 --dry-run
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PERSONAS_DIR = Path(__file__).parent
DEFINITIONS_DIR = PERSONAS_DIR / "definitions"
RUNS_DIR = PERSONAS_DIR / "runs"
EVAL_DIR = PERSONAS_DIR.parent  # evaluation/
PROJECT_ROOT = EVAL_DIR.parent


def load_persona(persona_name: str) -> dict:
    """Load and validate a persona YAML definition."""
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML required. Install with: pip install pyyaml")
        sys.exit(1)

    yaml_path = DEFINITIONS_DIR / f"{persona_name}.yaml"
    if not yaml_path.exists():
        print(f"ERROR: Persona definition not found: {yaml_path}")
        available = [f.stem for f in DEFINITIONS_DIR.glob("*.yaml")]
        if available:
            print(f"  Available personas: {', '.join(available)}")
        sys.exit(1)

    with open(yaml_path) as f:
        persona = yaml.safe_load(f)

    # Validate required fields
    required = ["persona", "research", "setup"]
    for field in required:
        if field not in persona:
            print(f"ERROR: Missing required field '{field}' in {yaml_path}")
            sys.exit(1)

    return persona


def get_next_version(persona_name: str) -> str:
    """Determine the next version number for a persona's runs."""
    run_dir = RUNS_DIR / persona_name
    if not run_dir.exists():
        return "v001"

    existing = sorted(d.name for d in run_dir.iterdir()
                      if d.is_dir() and d.name.startswith("v"))
    if not existing:
        return "v001"

    # Extract highest version number
    max_ver = 0
    for dirname in existing:
        try:
            ver_num = int(dirname.split("_")[0][1:])  # "v001_20260207" -> 1
            max_ver = max(max_ver, ver_num)
        except (ValueError, IndexError):
            continue

    return f"v{max_ver + 1:03d}"


def get_git_sha() -> str:
    """Get the current git SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        return result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def compute_config_hash(persona: dict) -> str:
    """Compute a hash of the persona's configuration for change detection."""
    config_str = json.dumps(persona, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(config_str.encode()).hexdigest()[:16]}"


def create_run_directory(persona_name: str, version: str) -> Path:
    """Create the versioned run directory structure."""
    date_str = datetime.now().strftime("%Y%m%d")
    run_name = f"{version}_{date_str}"
    run_dir = RUNS_DIR / persona_name / run_name

    # Create tier directories
    (run_dir / "tier1" / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "tier2").mkdir(parents=True, exist_ok=True)
    (run_dir / "tier3").mkdir(parents=True, exist_ok=True)

    return run_dir


def write_meta_json(run_dir: Path, persona: dict, version: str,
                    tier1_completed: bool = False,
                    checks_passed: int = 0, checks_total: int = 0,
                    duration_seconds: float = 0.0):
    """Write run metadata to meta.json."""
    meta = {
        "persona_id": persona["persona"]["id"],
        "persona_name": persona["persona"]["name"],
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "model": persona["setup"]["model"],
        "provider": persona["setup"]["provider"],
        "kosmos_git_sha": get_git_sha(),
        "config_hash": compute_config_hash(persona),
        "tier1_completed": tier1_completed,
        "tier2_completed": False,
        "tier3_completed": False,
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "duration_seconds": duration_seconds,
    }

    meta_path = run_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def run_tier1(persona_name: str, persona: dict, run_dir: Path, dry_run: bool = False):
    """Execute Tier 1: automated evaluation via scientific_evaluation.py."""
    eval_script = EVAL_DIR / "scientific_evaluation.py"
    if not eval_script.exists():
        print(f"ERROR: Evaluation script not found: {eval_script}")
        sys.exit(1)

    # Build command
    cmd = [
        sys.executable, str(eval_script),
        "--output-dir", str(run_dir / "tier1"),
    ]

    research = persona.get("research", {})
    if research.get("question"):
        cmd.extend(["--research-question", research["question"]])
    if research.get("domain"):
        cmd.extend(["--domain", research["domain"]])
    if research.get("dataset"):
        data_path = EVAL_DIR / research["dataset"]
        if data_path.exists():
            cmd.extend(["--data-path", str(data_path)])
        else:
            print(f"  WARNING: Dataset not found: {data_path}")
    if research.get("max_iterations"):
        cmd.extend(["--max-iterations", str(research["max_iterations"])])

    print(f"  Command: {' '.join(cmd)}")
    print()

    if dry_run:
        print("  [DRY RUN] Would execute the above command.")
        print(f"  [DRY RUN] Output would go to: {run_dir / 'tier1'}")
        return True

    # Clean persistent state for honest evaluation
    db_path = PROJECT_ROOT / "kosmos.db"
    cache_path = PROJECT_ROOT / ".kosmos_cache"
    if db_path.exists():
        db_path.unlink()
        print("  Cleared previous database for clean evaluation")
    if cache_path.exists():
        shutil.rmtree(cache_path)
        print("  Cleared previous caches for clean evaluation")

    # Execute
    print("  Running Tier 1 evaluation...")
    print("  " + "=" * 60)
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    print("  " + "=" * 60)

    if result.returncode != 0:
        print(f"  WARNING: Evaluation exited with code {result.returncode}")

    # Copy artifacts from default location if they exist and output-dir wasn't used
    # (backward compatibility: some artifacts may be written to evaluation/artifacts/)
    default_artifacts = EVAL_DIR / "artifacts"
    tier1_artifacts = run_dir / "tier1" / "artifacts"
    if default_artifacts.exists():
        for item in default_artifacts.iterdir():
            dest = tier1_artifacts / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

    # Copy the latest log file
    log_dir = EVAL_DIR / "logs"
    if log_dir.exists():
        logs = sorted(log_dir.glob("evaluation_*.log"))
        if logs:
            shutil.copy2(logs[-1], run_dir / "tier1" / "eval.log")

    return result.returncode == 0


def parse_tier1_results(run_dir: Path) -> tuple:
    """Parse Tier 1 results from the evaluation report."""
    report_path = run_dir / "tier1" / "EVALUATION_REPORT.md"
    checks_passed = 0
    checks_total = 0
    duration = 0.0

    if report_path.exists():
        content = report_path.read_text()
        # Parse "Checks passed: 36/37" or "36/37 checks passed" pattern
        import re
        match = re.search(r"Checks passed.*?(\d+)/(\d+)", content)
        if not match:
            match = re.search(r"(\d+)/(\d+)\s+checks?\s+passed", content)
        if match:
            checks_passed = int(match.group(1))
            checks_total = int(match.group(2))
        # Parse duration
        match = re.search(r"Duration.*?(\d+\.?\d*)\s*s", content)
        if match:
            duration = float(match.group(1))

    return checks_passed, checks_total, duration


def main():
    parser = argparse.ArgumentParser(
        description="Run persona-based scientific evaluation for Kosmos",
    )
    parser.add_argument(
        "--persona", required=True,
        help="Persona name (e.g., 001_enzyme_kinetics_biologist)",
    )
    parser.add_argument(
        "--tier", type=int, choices=[1], default=1,
        help="Tier to run (currently only Tier 1 is automated)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would execute without running",
    )
    parser.add_argument(
        "--version", type=str, default=None,
        help="Override version string (default: auto-increment)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  KOSMOS PERSONA EVALUATION")
    print("=" * 70)
    print()

    # Load persona
    persona = load_persona(args.persona)
    persona_info = persona["persona"]
    print(f"  Persona: {persona_info['name']} ({persona_info['role']})")
    print(f"  Question: {persona['research']['question']}")
    print(f"  Model: {persona['setup']['model']}")
    print()

    # Determine version
    version = args.version or get_next_version(args.persona)
    print(f"  Version: {version}")

    # Create run directory
    run_dir = create_run_directory(args.persona, version)
    print(f"  Run directory: {run_dir}")
    print()

    if args.dry_run:
        print("[DRY RUN MODE]")
        print()

    # Write initial meta.json
    write_meta_json(run_dir, persona, version)

    # Run Tier 1
    if args.tier >= 1:
        print("[Tier 1] Automated Evaluation")
        print("-" * 40)
        success = run_tier1(args.persona, persona, run_dir, dry_run=args.dry_run)

        if not args.dry_run:
            # Parse results and update meta.json
            checks_passed, checks_total, duration = parse_tier1_results(run_dir)
            write_meta_json(
                run_dir, persona, version,
                tier1_completed=success,
                checks_passed=checks_passed,
                checks_total=checks_total,
                duration_seconds=duration,
            )
            print()
            print(f"  Tier 1 result: {checks_passed}/{checks_total} checks passed")
        print()

    # Instructions for remaining tiers
    print("=" * 70)
    print("  NEXT STEPS")
    print("=" * 70)
    print()
    print("  Tier 2 (Technical Diagnostic):")
    print(f"    Have a Claude agent analyze {run_dir / 'tier1'}")
    print(f"    and write {run_dir / 'tier2' / 'TECHNICAL_REPORT.md'}")
    print()
    print("  Tier 3 (Narrative):")
    print(f"    Have a Claude agent write the persona narrative")
    print(f"    to {run_dir / 'tier3' / 'NARRATIVE.md'}")
    print()

    # Check for regression comparison
    persona_runs = RUNS_DIR / args.persona
    existing_versions = sorted(
        d.name for d in persona_runs.iterdir()
        if d.is_dir() and d.name.startswith("v") and d != run_dir
    )
    if existing_versions:
        prev = existing_versions[-1]
        current = run_dir.name
        print("  Regression Comparison:")
        print(f"    python evaluation/personas/compare_runs.py \\")
        print(f"      --persona {args.persona} \\")
        print(f"      --v1 {prev} --v2 {current}")
        print()


if __name__ == "__main__":
    main()

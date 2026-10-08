#!/usr/bin/env python3
"""Combine the 24-task, three-seed RoboCasa benchmark recorded in jobs.tsv."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

from monitor_robocasa_full_eval import SUBMISSIONS, read_manifest


TASKS = """
PnPCounterToCab PnPCabToCounter PnPCounterToSink PnPSinkToCounter
PnPCounterToMicrowave PnPMicrowaveToCounter PnPCounterToStove PnPStoveToCounter
OpenSingleDoor CloseSingleDoor OpenDoubleDoor CloseDoubleDoor
OpenDrawer CloseDrawer TurnOnStove TurnOffStove
TurnOnSinkFaucet TurnOffSinkFaucet TurnSinkSpout
CoffeeSetupMug CoffeeServeMug CoffeePressButton TurnOnMicrowave TurnOffMicrowave
""".split()
SEEDS = ("195", "196", "197")
TRIALS = 50


def parse_result(text, task):
    """Parse the final section; ignore rounded percentages and progress lines."""
    if "FINAL RESULTS" not in text:
        raise ValueError("No FINAL RESULTS section yet")
    if not re.search(r"^JOB_OK\s*$", text, re.MULTILINE):
        raise ValueError("Job did not finish cleanly (JOB_OK missing)")
    final = text.rsplit("FINAL RESULTS", 1)[1]

    def field(name, pattern):
        match = re.search(rf"^{re.escape(name)}:\s*({pattern})\s*$", final, re.MULTILINE)
        if not match:
            raise ValueError(f"Missing or invalid {name}")
        return match[1]

    reported_task = field("Task", r"\S+")
    if reported_task != task:
        raise ValueError(f"Task mismatch: expected {task}, found {reported_task}")
    episodes = int(field("Total episodes", r"\d+"))
    successes = int(field("Total successes", r"\d+"))
    length = float(field("Average episode length", r"[0-9]+(?:\.[0-9]+)?"))
    if episodes != TRIALS:
        raise ValueError(f"Expected {TRIALS} episodes, found {episodes}")
    if not 0 <= successes <= episodes or not math.isfinite(length):
        raise ValueError("Invalid success count or episode length")
    result_path = re.search(r"^Results saved to:\s*(.+)$", final, re.MULTILINE)
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate_percent": 100 * successes / episodes,
        "average_episode_length": length,
        "result_file": result_path[1].strip() if result_path else "",
    }


def collect(jobs, log_dir):
    by_pair = {}
    for job_id, (task, seed) in jobs.items():
        if task not in TASKS or seed not in SEEDS:
            raise RuntimeError(f"Unexpected task/seed in manifest: {task}, {seed}")
        pair = task, seed
        if pair in by_pair:
            raise RuntimeError(f"Duplicate {task}, seed {seed}; select a manifest containing one benchmark batch")
        by_pair[pair] = job_id
    rows = []
    for task in TASKS:
        for seed in SEEDS:
            job_id = by_pair.get((task, seed), "")
            log_file = log_dir / f"spatial-gpu-{job_id}.out" if job_id else None
            row = {
                "task": task, "seed": seed, "job_id": job_id,
                "status": "MISSING_SUBMISSION", "detail": "No job recorded for this task/seed",
                "episodes": None, "successes": None, "success_rate_percent": None,
                "average_episode_length": None, "result_file": "",
                "stdout_file": str(log_file) if log_file else "",
            }
            if log_file:
                try:
                    result = parse_result(log_file.read_text(errors="replace"), task)
                    row.update(result, status="COMPLETE", detail="")
                except FileNotFoundError:
                    row.update(status="MISSING_LOG", detail="Job stdout does not exist yet")
                except ValueError as exc:
                    row.update(status="INCOMPLETE", detail=str(exc))
            rows.append(row)
    return rows


def aggregate(rows, manifest, submission):
    valid = [row for row in rows if row["status"] == "COMPLETE"]
    complete = len(valid) == len(TASKS) * len(SEEDS) and submission == "COMPLETE"
    seed_results = []
    for seed in SEEDS:
        selected = [row for row in valid if row["seed"] == seed]
        episodes = sum(row["episodes"] for row in selected)
        successes = sum(row["successes"] for row in selected)
        seed_results.append({
            "seed": seed, "completed_runs": len(selected), "episodes": episodes,
            "successes": successes,
            "success_rate_percent": 100 * successes / episodes if len(selected) == len(TASKS) else None,
        })
    task_results = []
    for task in TASKS:
        selected = [row for row in valid if row["task"] == task]
        episodes = sum(row["episodes"] for row in selected)
        successes = sum(row["successes"] for row in selected)
        entry = {
            "task": task, "completed_runs": len(selected), "episodes": episodes,
            "successes": successes,
            "success_rate_percent": 100 * successes / episodes if len(selected) == len(SEEDS) else None,
        }
        for seed in SEEDS:
            entry[f"seed_{seed}_percent"] = next(
                (row["success_rate_percent"] for row in selected if row["seed"] == seed), None
            )
        task_results.append(entry)
    overall = None
    if complete:
        successes = sum(row["successes"] for row in valid)
        episodes = sum(row["episodes"] for row in valid)
        overall = {
            "successes": successes, "episodes": episodes,
            "success_rate_percent": 100 * successes / episodes,
            "seed_mean_percent": statistics.mean(row["success_rate_percent"] for row in seed_results),
            "seed_sample_std_percentage_points": statistics.stdev(row["success_rate_percent"] for row in seed_results),
        }
    return {
        "manifest": str(manifest.resolve()), "submission_state": submission,
        "complete": complete, "expected_runs": len(TASKS) * len(SEEDS),
        "completed_runs": len(valid), "expected_episodes": len(TASKS) * len(SEEDS) * TRIALS,
        "overall": overall, "by_seed": seed_results, "by_task": task_results, "runs": rows,
    }


def report_text(report):
    lines = [f"Completed evaluations: {report['completed_runs']}/{report['expected_runs']}"]
    if report["complete"]:
        overall = report["overall"]
        lines += [
            f"Overall success rate: {overall['success_rate_percent']:.2f}% ({overall['successes']}/{overall['episodes']} episodes)",
            f"Mean across seeds: {overall['seed_mean_percent']:.2f}%",
            f"Sample standard deviation across seeds: {overall['seed_sample_std_percentage_points']:.2f} percentage points",
            "", "By seed:",
        ]
        for row in report["by_seed"]:
            lines.append(f"  {row['seed']}: {row['success_rate_percent']:.2f}% ({row['successes']}/{row['episodes']})")
        lines += ["", f"{'TASK':<26} {'195 %':>7} {'196 %':>7} {'197 %':>7} {'MEAN %':>8}"]
        for row in report["by_task"]:
            lines.append(f"{row['task']:<26} {row['seed_195_percent']:7.2f} {row['seed_196_percent']:7.2f} {row['seed_197_percent']:7.2f} {row['success_rate_percent']:8.2f}")
    else:
        lines += ["INCOMPLETE: final benchmark score withheld.", f"Submission state: {report['submission_state']}"]
        for row in report["runs"]:
            if row["status"] != "COMPLETE":
                lines.append(f"  {row['task']}, seed {row['seed']}, job {row['job_id'] or '-'}: {row['detail']}")
    return "\n".join(lines) + "\n"


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, help="jobs.tsv (default: latest recorded batch)")
    parser.add_argument("--log-dir", type=Path, default=Path("/spatial-storage") / os.environ.get("USER", "") / "logs", help="Directory containing spatial-gpu-JOB_ID.out")
    parser.add_argument("--output-dir", type=Path, help="Report destination (default: manifest's directory)")
    args = parser.parse_args()
    manifest = args.manifest
    if manifest is None:
        candidates = list(SUBMISSIONS.glob("*/jobs.tsv"))
        if not candidates:
            parser.error("No jobs.tsv found; run the batch monitor once to recover existing submissions")
        manifest = max(candidates, key=lambda path: path.parent.name)
    jobs, submission = read_manifest(manifest)
    report = aggregate(collect(jobs, args.log_dir), manifest, submission)
    destination = args.output_dir or manifest.parent
    destination.mkdir(parents=True, exist_ok=True)
    text = report_text(report)
    (destination / "summary.txt").write_text(text)
    (destination / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_csv(destination / "results.csv", report["runs"])
    write_csv(destination / "task_results.csv", report["by_task"])
    print(text, end="")
    print(f"Reports saved to: {destination.resolve()}")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError) as exc:
        print(f"summarize-full-eval: {exc}", file=sys.stderr)
        sys.exit(2)

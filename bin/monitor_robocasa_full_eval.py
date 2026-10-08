#!/usr/bin/env python3
"""Watch a recorded RoboCasa SLURM batch without cancelling any jobs."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parent.parent
SUBMISSIONS = PROJECT / "results/robocasa-predict2.5/full-eval/submissions"
TERMINAL = {
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL",
    "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT",
}


def query(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"{command[0]}: {result.stderr.strip() or 'query failed'}")
    return result.stdout


def discover(since):
    """Recover legacy submissions using their recorded arguments, not job order."""
    output = query([
        "sacct", "--user", os.environ.get("USER", ""), "--starttime", since,
        "--noheader", "--parsable2", "--format=JobID%100,SubmitLine%4096",
    ])
    jobs = {}
    root = PROJECT / "results/robocasa-predict2.5/full-eval"
    for line in output.splitlines():
        job_id, separator, command = line.partition("|")
        job_id = job_id.strip()
        if not separator or not re.fullmatch(r"\d+", job_id):
            continue
        try:
            words = shlex.split(command.rstrip("|"))
            if not any(Path(word).name == "run_robocasa_predict2p5.py" for word in words):
                continue
            task = words[words.index("--task_name") + 1]
            seed = words[words.index("--seed") + 1]
            note = words[words.index("--run_id_note") + 1]
            log_dir = Path(words[words.index("--local_log_dir") + 1])
            if note != f"full-eval-seed-{seed}" or log_dir != root / task / f"seed-{seed}":
                continue
        except (ValueError, IndexError):
            continue
        jobs[job_id] = (task, seed)
    if not jobs:
        raise RuntimeError(f"No full-evaluation submissions found since {since}.")
    directory = SUBMISSIONS / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-recovered-{os.getpid()}")
    directory.mkdir(parents=True)
    manifest = directory / "jobs.tsv"
    with manifest.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("job_id", "task", "seed"))
        for job_id, (task, seed) in sorted(jobs.items(), key=lambda item: int(item[0])):
            writer.writerow((job_id, task, seed))
    (directory / "submission.state").write_text("COMPLETE\n")
    print(f"Recovered {len(jobs)} jobs. Manifest: {manifest}", flush=True)
    return manifest


def read_manifest(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["job_id", "task", "seed"]:
            raise RuntimeError(f"Invalid manifest header: {path}")
        rows = list(reader)
    jobs = {}
    for row in rows:
        # A concurrent append can briefly expose an unfinished final line.
        if not all(row.values()):
            continue
        if not re.fullmatch(r"\d+", row["job_id"]) or row["job_id"] in jobs:
            raise RuntimeError(f"Invalid or duplicate job ID in {path}")
        jobs[row["job_id"]] = (row["task"], row["seed"])
    state_file = path.parent / "submission.state"
    submission = (state_file.read_text().strip() or "SUBMITTING") if state_file.exists() else "COMPLETE"
    if submission not in {"SUBMITTING", "COMPLETE", "FAILED"}:
        raise RuntimeError(f"Invalid submission state in {state_file}: {submission}")
    return jobs, submission


def statuses(jobs):
    if not jobs:
        return {}
    ids = ",".join(jobs)
    # squeue may reject valid completed IDs, which remain available in sacct.
    live = subprocess.run(
        ["squeue", "--jobs", ids, "--noheader", "--format=%i|%T|%M|%R"],
        capture_output=True, text=True, timeout=30,
    )
    accounting = query([
        "sacct", "--jobs", ids, "--noheader", "--parsable2",
        "--format=JobID%100,State%100,ExitCode,Elapsed,NodeList%100",
    ])
    if live.returncode and "Invalid job id" not in live.stderr:
        raise RuntimeError(f"squeue: {live.stderr.strip() or 'query failed'}")
    records = {}
    for line in accounting.splitlines():
        fields = [field.strip() for field in line.split("|")]
        if len(fields) >= 5 and fields[0] in jobs:
            state = fields[1].split()[0].rstrip("+") if fields[1] else "UNKNOWN"
            records[fields[0]] = (state, fields[3], fields[2], fields[4])
    for line in live.stdout.splitlines():
        fields = [field.strip() for field in line.split("|")]
        if len(fields) == 4 and fields[0] in jobs:
            # Prefer current scheduler state over potentially delayed accounting.
            code = records.get(fields[0], ("", "", "-", ""))[2] if fields[1] in TERMINAL else "-"
            records[fields[0]] = (fields[1], fields[2], code, fields[3])
    return {job_id: records.get(job_id, ("WAITING_ACCOUNTING", "-", "-", "-")) for job_id in jobs}


def successful(record):
    return record[0] == "COMPLETED" and record[2] == "0:0"


def display(manifest, jobs, submission, records, clear):
    if clear:
        print("\033[2J\033[H", end="")
    counts = Counter(record[0] for record in records.values())
    done = sum(count for state, count in counts.items() if state in TERMINAL)
    print(f"RoboCasa full evaluation — {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}")
    print(f"Manifest: {manifest}")
    print(f"Submission: {submission} | Finished: {done}/{len(jobs)} | " +
          ", ".join(f"{state}: {count}" for state, count in sorted(counts.items())))
    print(f"{'JOB ID':<10} {'TASK':<26} {'SEED':<5} {'STATE':<19} {'ELAPSED':<12} {'EXIT':<7} NODE / PENDING REASON")
    for job_id, (task, seed) in jobs.items():
        state, elapsed, code, detail = records[job_id]
        print(f"{job_id:<10} {task:<26} {seed:<5} {state:<19} {elapsed:<12} {code:<7} {detail}")
    print("Ctrl-C detaches; jobs continue in SLURM.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, help="jobs.tsv (default: latest recorded batch)")
    parser.add_argument("--discover", action="store_true", help="Recover existing full-evaluation jobs from accounting")
    parser.add_argument("--since", default="today", help="Accounting start time for recovery (default: today)")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    parser.add_argument("--interval", type=float, default=os.environ.get("POLL_INTERVAL", "10"), help="Refresh seconds (default: POLL_INTERVAL or 10)")
    args = parser.parse_args()
    if args.manifest and args.discover:
        parser.error("Choose either a manifest or --discover")
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be a positive finite number")
    for command in ("squeue", "sacct"):
        if not shutil.which(command):
            parser.error(f"Required SLURM command not found: {command}")
    candidates = list(SUBMISSIONS.glob("*/jobs.tsv"))
    manifest = args.manifest
    if manifest is None:
        if args.discover or not candidates:
            manifest = discover(args.since)
        else:
            manifest = max(candidates, key=lambda path: path.parent.name)
    print(f"Monitoring: {manifest}", flush=True)
    previous = None
    while True:
        jobs, submission = read_manifest(manifest)
        try:
            records = statuses(jobs)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            if args.once:
                raise
            print(f"Status query failed; retrying in {args.interval:g}s: {exc}", file=sys.stderr, flush=True)
            time.sleep(args.interval)
            continue
        snapshot = (submission, records)
        if sys.stdout.isatty() or snapshot != previous or args.once:
            display(manifest, jobs, submission, records, sys.stdout.isatty() and not args.once)
            previous = snapshot
        finished = submission != "SUBMITTING" and all(
            record[0] in TERMINAL and record[2] not in {"", "-"}
            for record in records.values()
        )
        if finished:
            failures = [job_id for job_id, record in records.items() if not successful(record)]
            print(f"Batch finished: {len(jobs) - len(failures)}/{len(jobs)} jobs completed successfully.")
            if failures:
                print("Failed job IDs: " + ", ".join(failures))
            if submission == "FAILED":
                print("Submission stopped early; this manifest contains only the accepted jobs.")
            return 1 if failures or submission == "FAILED" else 0
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nDetached from batch; jobs continue in SLURM.", file=sys.stderr)
        sys.exit(130)
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"monitor-full-eval: {exc}", file=sys.stderr)
        sys.exit(2)

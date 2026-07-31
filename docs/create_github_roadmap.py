#!/usr/bin/env python3
"""
Turns projects.csv into a live GitHub roadmap:
  - one Milestone per stage (deduped, due date = last item's target date in that stage)
  - one Issue per project row, labeled by track/difficulty/priority, assigned to its milestone
  - each issue added as an item to a GitHub Project (Projects v2), for the Roadmap view

Requirements:
  - GitHub CLI installed and authenticated: https://cli.github.com  ->  gh auth login
  - A repo to create issues in (owner/repo)
  - A GitHub Project (v2) already created, with its project number
    (Project number is the number in the project's URL, e.g. .../projects/3 -> 3)

Usage:
  python3 create_github_roadmap.py --repo YOUR_USER/AI-Portfolio --project-owner YOUR_USER --project-number 1
  python3 create_github_roadmap.py --repo YOUR_USER/AI-Portfolio --project-owner YOUR_USER --project-number 1 --execute

By default this is a DRY RUN: it only prints the gh commands it would run.
Pass --execute to actually create labels, milestones, issues, and project items.
"""
import argparse
import csv
import subprocess
import sys

TRACK_COLORS = {
    "Data Science": "0e8a16",
    "Machine Learning": "1d76db",
    "Deep Learning": "5319e7",
    "Computer Vision": "b60205",
    "Generative AI": "e99695",
    "LLM Systems": "fbca04",
    "Production AI": "0052cc",
    "Program": "6a737d",
}


def run(cmd, execute):
    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    print(f"$ {printable}")
    if not execute:
        return ""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ! non-fatal error: {result.stderr.strip()}", file=sys.stderr)
    return result.stdout.strip()


def ensure_label(repo, name, color, execute):
    run(["gh", "label", "create", name, "--repo", repo, "--color", color, "--force"], execute)


def ensure_milestone(repo, title, due_on, execute):
    # gh has no native `milestone create`; use the REST API via gh api.
    run([
        "gh", "api", f"repos/{repo}/milestones", "-f", f"title={title}",
        "-f", f"due_on={due_on}T00:00:00Z", "-f", "state=open",
    ], execute)


def create_issue(repo, title, body, labels, milestone, execute):
    cmd = ["gh", "issue", "create", "--repo", repo, "--title", title, "--body", body]
    for lab in labels:
        cmd += ["--label", lab]
    if milestone:
        cmd += ["--milestone", milestone]
    return run(cmd, execute)  # stdout is the issue URL when --execute is set


def add_to_project(project_owner, project_number, issue_url, execute):
    run([
        "gh", "project", "item-add", str(project_number),
        "--owner", project_owner, "--url", issue_url,
    ], execute)


def build_body(row):
    lines = [f"**Track:** {row['Track']}", f"**Stage:** {row['Stage']}", f"**Month:** {row['Month']} (target {row['TargetDate']})"]
    if row["Difficulty"]:
        lines.append(f"**Difficulty:** {row['Difficulty']}")
    if row["Priority"]:
        lines.append(f"**Priority:** {row['Priority']}")
    if row["Domain"]:
        lines.append(f"**Domain:** {row['Domain']}")
    lines += [
        "",
        "### Deliverables",
        "- [ ] Research artifact — notebook, experiment report",
        "- [ ] Engineering artifact — API, pipeline, Docker",
        "- [ ] User artifact — dashboard or application",
        "- [ ] Communication artifact — blog post, LinkedIn post, demo video",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="projects.csv")
    ap.add_argument("--repo", required=True, help="owner/repo to create issues in")
    ap.add_argument("--project-owner", required=True, help="user or org that owns the GitHub Project")
    ap.add_argument("--project-number", required=True, type=int, help="number from the project URL")
    ap.add_argument("--execute", action="store_true", help="actually run the commands (default is dry-run/print-only)")
    args = ap.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not args.execute:
        print("### DRY RUN — no changes will be made. Pass --execute to apply. ###\n")

    # 1. Labels: one per track, one per difficulty tier, one for flagship priority
    tracks = sorted({r["Track"] for r in rows if r["Track"]})
    difficulties = sorted({r["Difficulty"] for r in rows if r["Difficulty"]})
    print("== Creating labels ==")
    for t in tracks:
        ensure_label(args.repo, f"track:{t}", TRACK_COLORS.get(t, "cccccc"), args.execute)
    for d in difficulties:
        ensure_label(args.repo, f"level:{d}", "c2e0c6", args.execute)
    ensure_label(args.repo, "flagship", "ffd700", args.execute)
    ensure_label(args.repo, "milestone-item", "6a737d", args.execute)

    # 2. Milestones: one per Stage, due date = latest TargetDate seen for that stage
    print("\n== Creating milestones ==")
    stage_due = {}
    for r in rows:
        stage_due[r["Stage"]] = max(stage_due.get(r["Stage"], ""), r["TargetDate"])
    for stage, due in stage_due.items():
        ensure_milestone(args.repo, stage, due, args.execute)

    # 3. Issues + project items
    print("\n== Creating issues and adding to project ==")
    for r in rows:
        is_project = r["Type"] == "Project"
        prefix = f"Project {r['Number']} — " if is_project else "Milestone — "
        title = f"{prefix}{r['Title']}"

        labels = [f"track:{r['Track']}"]
        if r["Difficulty"]:
            labels.append(f"level:{r['Difficulty']}")
        if "Flagship" in r["Priority"]:
            labels.append("flagship")
        if not is_project:
            labels.append("milestone-item")

        body = build_body(r)
        url = create_issue(args.repo, title, body, labels, r["Stage"], args.execute)
        if args.execute and url:
            add_to_project(args.project_owner, args.project_number, url, args.execute)

    print("\nDone." if args.execute else "\nDry run complete — re-run with --execute to apply.")


if __name__ == "__main__":
    main()

"""Download PR data from GitHub via gh CLI."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from datetime import date, timedelta
from pathlib import Path

import logfire
import typer

from braindump.config import RepoConfig
from braindump.progress import StageStats, console, stage_progress

_PR_LIST_FIELDS = (
    "number,title,author,state,createdAt,updatedAt,mergedAt,closedAt,baseRefName,headRefName,url"
)

# The GitHub search API, used by `gh pr list --search`, only returns the first 1000 results.
_SEARCH_RESULT_CAP = 1000

# Chunks at least this large are cross-checked against the search API's total_count, because a
# truncated result set can come back below the cap (observed on django/django: 999 of 16449).
_VERIFY_FROM = 300

# Per-PR resources that can be downloaded: name -> (subdirectory, file extension, span label).
# Only review_comments is read by later pipeline stages; the rest exist for status reporting and
# future use. `--resources` selects a subset, `all` downloads every resource.
RESOURCES: dict[str, tuple[str, str, str]] = {
    "prs": ("prs", "json", "PR details"),
    "reviews": ("reviews", "json", "reviews"),
    "review_comments": ("review_comments", "json", "review comments"),
    "diffs": ("diffs", "diff", "diff"),
    "issue_comments": ("issue_comments", "json", "issue comments"),
}

ALL_RESOURCES = ",".join(RESOURCES)


def _parse_resources(resources: str) -> list[str]:
    """Parse a `--resources` value into resource names ('all' selects every resource)."""
    if resources.strip().lower() == "all":
        return list(RESOURCES)
    names = [name.strip() for name in resources.split(",") if name.strip()]
    unknown = [name for name in names if name not in RESOURCES]
    if not names or unknown:
        console.print(
            f"[red]Error: --resources must be 'all' or a comma-separated subset of "
            f"{ALL_RESOURCES} (got: {resources!r})[/red]"
        )
        raise typer.Exit(1)
    return names


def _resource_args(name: str, repo: str, pr_num: int) -> list[str]:
    """Build the read-only `gh api` arguments for one resource of one PR."""
    if name == "prs":
        return ["api", f"repos/{repo}/pulls/{pr_num}"]
    if name == "diffs":
        return [
            "api",
            f"repos/{repo}/pulls/{pr_num}",
            "-H",
            "Accept: application/vnd.github.v3.diff",
        ]
    if name == "reviews":
        return ["api", f"repos/{repo}/pulls/{pr_num}/reviews", "--paginate"]
    if name == "review_comments":
        return ["api", f"repos/{repo}/pulls/{pr_num}/comments", "--paginate"]
    return ["api", f"repos/{repo}/issues/{pr_num}/comments", "--paginate"]


def requested_resources(output_dir: Path) -> list[str]:
    """Read the resource selection persisted by the download stage (all resources if unset)."""
    path = output_dir / "resources.json"
    if path.exists():
        names = json.loads(path.read_text()).get("resources", [])
        if names and all(name in RESOURCES for name in names):
            return names
    return list(RESOURCES)


def _fetch_prs(repo: str, since: str | None) -> list[dict]:
    """Fetch the PRs updated since `since` (all PRs if None), newest first.

    A range whose query comes back at (or near) the search result cap is split in half and
    re-fetched, recursively, until every chunk is complete. Chunks are half-open ([lo, hi)) so they
    never overlap or leave gaps.
    """
    lo0 = date.fromisoformat(since) if since else date(2000, 1, 1)
    hi0 = date.today() + timedelta(days=1)

    def search_total(lo: date, hi: date) -> int:
        """Number of PRs matching an updated-range query (search API's total_count, -1 on error).

        `gh pr list --search` can come back truncated *below* its 1000-result cap (observed: a
        full-history query returning 999 of 16449 matches), so the fetched count alone cannot
        prove a chunk complete. total_count is authoritative.
        """
        q = f"repo:{repo} is:pr updated:>={lo.isoformat()} updated:<{hi.isoformat()}"
        for attempt in range(2):
            result = _gh(
                "api",
                "-X",
                "GET",
                "search/issues",
                "-f",
                f"q={q}",
                "--jq",
                ".total_count",
                check=False,
            )
            try:
                return int(result.stdout.strip())
            except ValueError:
                if attempt == 0:
                    time.sleep(10)
        logfire.warn("Could not read search total_count", lo=lo.isoformat(), hi=hi.isoformat())
        return -1

    def fetch(lo: date, hi: date, depth: int = 0) -> list[dict]:
        args = [
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "all",
            "--limit",
            str(_SEARCH_RESULT_CAP),
            "--json",
            _PR_LIST_FIELDS,
            "--search",
            f"updated:>={lo.isoformat()} updated:<{hi.isoformat()}",
        ]
        prs = json.loads(_gh(*args).stdout)
        complete = len(prs) < _SEARCH_RESULT_CAP
        if complete and len(prs) >= _VERIFY_FROM:
            total = search_total(lo, hi)
            # Unverifiable chunks are split rather than trusted: a silently truncated PR list
            # makes the download stage report completion it never reached.
            complete = total >= 0 and len(prs) >= total
        if complete:
            return prs
        if depth >= 8 or (hi - lo).days <= 1:
            logfire.warn(
                "PR list chunk is still incomplete after splitting",
                lo=lo.isoformat(),
                hi=hi.isoformat(),
                count=len(prs),
            )
            return prs
        mid = lo + (hi - lo) / 2
        return fetch(lo, mid, depth + 1) + fetch(mid, hi, depth + 1)

    prs = fetch(lo0, hi0)
    prs.sort(key=lambda pr: pr["updatedAt"], reverse=True)
    logfire.info("Fetched PR list", total=len(prs), since=since)
    return prs


def _gh(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a gh CLI command (sync, for auth check and PR list)."""
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=check)


async def _gh_async(*args: str) -> tuple[int, str, str]:
    """Run a gh CLI command asynchronously. Returns (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        "gh",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode or 0, stdout.decode(), stderr.decode()


@logfire.instrument("download", extract_args={"config": ["repo"]})
def _run(
    config: RepoConfig,
    since: str | None = None,
    concurrency: int = 5,
    is_pipeline: bool = False,
    fresh: bool = False,
    resources: str = "all",
) -> StageStats | None:
    """Download PR reviews for a repo."""
    import shutil

    repo = config.repo
    output_dir = config.pr_data_dir
    resource_names = _parse_resources(resources)

    # Check gh auth
    result = _gh("auth", "status", check=False)
    if result.returncode != 0:
        console.print("[red]Error: gh CLI is not authenticated. Run 'gh auth login' first.[/red]")
        raise typer.Exit(1)

    if fresh and output_dir.exists():
        shutil.rmtree(output_dir)

    # Create directories for the requested resources and record the selection for `status`
    for subdir, _, _label in (RESOURCES[name] for name in resource_names):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    (output_dir / "resources.json").write_text(json.dumps({"resources": resource_names}, indent=2))

    if not is_pipeline:
        if since:
            console.print(f"Fetching PRs from {repo} since {since}...")
        else:
            console.print(f"Fetching all PRs from {repo}...")

    # Get PR list (sync — runs once)
    with logfire.span("fetch PR list"):
        prs = _fetch_prs(repo, since)
        pr_list_path = output_dir / "pr_list.json"
        pr_list_path.write_text(json.dumps(prs, indent=2))

    pr_numbers = [pr["number"] for pr in prs]
    logfire.info("Found PRs to download", total=len(pr_numbers), since=since)

    return asyncio.run(
        _async_download(repo, output_dir, pr_numbers, concurrency, is_pipeline, resource_names)
    )


async def _async_download(
    repo: str,
    output_dir,
    pr_numbers: list[int],
    concurrency: int,
    is_pipeline: bool,
    resources: list[str],
) -> StageStats | None:
    """Download all requested PR data in parallel using asyncio."""
    semaphore = asyncio.Semaphore(concurrency)

    def _pr_already_downloaded(pr_num: int) -> bool:
        """Check if every requested resource file exists for a PR."""
        return all(
            (output_dir / RESOURCES[name][0] / f"{pr_num}.{RESOURCES[name][1]}").exists()
            for name in resources
        )

    already_done = [pr for pr in pr_numbers if _pr_already_downloaded(pr)]
    remaining = [pr for pr in pr_numbers if not _pr_already_downloaded(pr)]

    async def download_pr(pr_num: int) -> None:
        """Download the requested resources for a single PR."""
        async with semaphore:
            with logfire.span("download PR #{pr_num}", pr_num=pr_num):
                for name in resources:
                    subdir, ext, label = RESOURCES[name]
                    with logfire.span(f"fetch {label}: #{pr_num}", pr_num=pr_num):
                        rc, stdout, stderr = await _gh_async(*_resource_args(name, repo, pr_num))
                        if rc == 0:
                            (output_dir / subdir / f"{pr_num}.{ext}").write_text(stdout)
                        elif "403" in stderr or "429" in stderr:
                            logfire.warn(
                                f"Rate limited fetching {label}",
                                pr_num=pr_num,
                                stderr=stderr[:200],
                            )

    with stage_progress("Download", is_pipeline=is_pipeline) as (progress, stats):
        stats.set("PRs", len(pr_numbers))
        if not is_pipeline:
            console.print(
                f"Downloading {len(remaining)} PRs (already downloaded: {len(already_done)})"
            )

        task = progress.add_task(
            "Downloading PR data...", total=len(pr_numbers), completed=len(already_done)
        )

        async def download_with_progress(pr_num: int) -> None:
            await download_pr(pr_num)
            progress.advance(task)

        await asyncio.gather(*[download_with_progress(pr_num) for pr_num in remaining])

        stats.detail = f"{len(pr_numbers)} PRs"
        return stats


def download(
    ctx: typer.Context,
    since: str | None = typer.Option(
        None, help="Only fetch PRs updated since this date (YYYY-MM-DD)"
    ),
    concurrency: int = typer.Option(5, help="Number of parallel PR downloads"),
    resources: str = typer.Option(
        "all",
        help=f"Comma-separated resources to fetch, or 'all' ({ALL_RESOURCES})",
    ),
    fresh: bool = typer.Option(False, "--fresh", help="Wipe output and start from scratch"),
) -> None:
    """Download PR data from GitHub via gh CLI."""
    config: RepoConfig = ctx.obj["config"]
    _run(config, since=since, concurrency=concurrency, fresh=fresh, resources=resources)

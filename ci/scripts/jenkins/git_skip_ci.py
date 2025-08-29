import argparse
import logging
import os

from cmd_utils import init_log, tags_from_title
from git_utils import GitHubRepo, git, parse_remote

if __name__ == "__main__":
    help = "Exits with 0 if CI should be skipped, 1 otherwise"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--pr", required=True)
    parser.add_argument("--remote", default="origin", help="ssh remote to parse")
    parser.add_argument(
        "--pr-title", help="(testing) PR title to use instead of fetching from GitHub"
    )
    args = parser.parse_args()
    init_log()
    branch = git(["rev-parse", "--abbrev-ref", "HEAD"])
    log = git(["log", "--format=%s", "-1"])

    def check_pr_title():
        remote = git(["config", "--get", f"remote.{args.remote}.url"])
        user, repo = parse_remote(remote)
        if args.pr_title:
            title = args.pr_title
        else:
            github = GitHubRepo(token=os.environ["GITHUB_TOKEN"], user=user, repo=repo)
            pr = github.get(f"pulls/{args.pr}")
            title = pr["title"]
        logging.info(f"pr title: {title}")
        tags = tags_from_title(title)
        logging.info(f"Found title tags: {tags}")
        return "skip ci" in tags

    if (
        args.pr != "null"
        and args.pr.strip() != ""
        and branch != "main"
        and check_pr_title()
    ):
        logging.info("PR title starts with '[skip ci]', skipping...")
        exit(0)
    else:
        logging.info(
            f"Not skipping CI:\nargs.pr: {args.pr}\nbranch: {branch}\ncommit: {log}"
        )
        exit(1)

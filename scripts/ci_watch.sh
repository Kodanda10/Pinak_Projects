#!/usr/bin/env bash
set -euo pipefail

# Simple CI watcher via GitHub API.
# Requires GH_CI_TOKEN with repo + workflow read on this repo.

if [[ -z "${GH_CI_TOKEN:-}" ]]; then
  echo "Set GH_CI_TOKEN to a GitHub token with workflow read access." >&2
  exit 2
fi

REPO_URL=$(git config --get remote.origin.url)
REPO=${REPO_URL#*:}
REPO=${REPO%.git}
OWNER=${REPO%%/*}
NAME=${REPO##*/}

api() {
  curl -sS -H "Authorization: Bearer ${GH_CI_TOKEN}" -H "Accept: application/vnd.github+json" "https://api.github.com/repos/${OWNER}/${NAME}/$1"
}

echo "Latest workflow runs:"
api actions/runs?per_page=10 | jq '.workflow_runs[] | {name: .name, status: .status, conclusion: .conclusion, url: .html_url}'

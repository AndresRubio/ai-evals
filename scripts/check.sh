#!/usr/bin/env bash
# Run every check CI runs (.github/workflows/ci.yml), for a branch that has not been
# pushed. The Python 3.13 tests get their own environment, .venv-3.13, so .venv stays
# on 3.14. Every step runs even after one fails, and the failed ones are listed last.
set -uo pipefail
cd "$(dirname "$0")/.."

# GitHub Actions sets GITHUB_ACTIONS=true, and Typer then renders CLI output in color,
# so a test can pass here and fail there. CI=true stays unset: it turns a skipped root
# or locale test into a failure, and those skip on a laptop.
export GITHUB_ACTIONS=true

failed=()
step() {
  echo "==> $*"
  "$@" || failed+=("$*")
}

step uv run ruff check .
step uv run mypy src/axial
step uv run pytest -q
step env UV_PROJECT_ENVIRONMENT=.venv-3.13 uv run --python 3.13 pytest -q
step uv run python scripts/knowledge_index.py --check

if ((${#failed[@]})); then
  echo "==> failed:"
  printf '    %s\n' "${failed[@]}"
  exit 1
fi
echo "==> all checks passed"

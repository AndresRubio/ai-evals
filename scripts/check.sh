#!/usr/bin/env bash
# Run every check CI runs (.github/workflows/ci.yml), for a branch that has not been
# pushed. The Python 3.13 tests get their own environment, .venv-3.13, so .venv stays
# on 3.14. Every step runs even after one fails, and the failed ones are listed last.
#
# A pass is recorded against the hash of the working tree, in the clone's git directory,
# so .githooks/pre-push can refuse a commit whose tree never passed. A second run on a tree
# that passed stops early, and --force runs the checks anyway.
set -uo pipefail
cd "$(dirname "$0")/.."

force=
case "${1:-}" in
  "") ;;
  --force) force=1 ;;
  *)
    echo "usage: scripts/check.sh [--force]" >&2
    exit 2
    ;;
esac

# The tree a commit of everything here would have, ignored files aside. It starts from a
# copy of the index, which a tracked file matching .gitignore needs to stay in, and leaves
# the real index alone. The copy keeps the index's timestamp, because git compares it with
# a file's to notice an edit of the same size made in the same second. A repository nested
# here that the index does not hold, as it holds a submodule, would enter the tree as a link
# no commit here has, so that fails too. Outside a git checkout it prints nothing and fails.
nested_repositories() { git ls-files -s | awk '$1 == 160000' | cut -f2 | sort; }
working_tree() {
  local dir index status=1
  index="$(git rev-parse --git-path index 2>/dev/null)" || return 1
  dir="$(mktemp -d "${TMPDIR:-/tmp}/axial-check.XXXXXX")" || return 1
  [[ -f "$index" ]] && cp -p "$index" "$dir/index"
  if GIT_INDEX_FILE="$dir/index" git -c advice.addEmbeddedRepo=false add -A; then
    if [[ -z "$(comm -13 <(nested_repositories) <(GIT_INDEX_FILE="$dir/index" nested_repositories))" ]]; then
      GIT_INDEX_FILE="$dir/index" git write-tree
      status=$?
    else
      echo "==> a git repository sits inside this one; ignore it or move it, or no pass can count" >&2
    fi
  fi
  rm -rf "$dir"
  return $status
}

# The hook in effect is compared with this one, as core.hooksPath has many spellings. Worktrees
# of a clone share that setting, so one clone needs it once.
remind_about_the_hook() {
  local hook
  [[ -n "$git_dir" ]] || return 0
  hook="$(git rev-parse --git-path hooks/pre-push)"
  if ! { [[ -x "$hook" ]] && cmp -s "$hook" .githooks/pre-push; }; then
    echo "==> the pre-push hook is off; turn it on with: git config core.hooksPath .githooks"
  fi
}

tree="$(working_tree)" || tree=
git_dir="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" || git_dir=
stamps="$git_dir/axial-checked"

if [[ -z "$force" && -n "$tree" && -e "$stamps/$tree" ]]; then
  echo "==> already checked: this exact tree passed; --force runs the checks again"
  remind_about_the_hook
  exit 0
fi

# GitHub Actions sets GITHUB_ACTIONS=true, and Typer then renders CLI output in color,
# so a test can pass here and fail there. CI=true stays unset: it turns a skipped root
# or locale test into a failure, and those skip on a laptop.
export GITHUB_ACTIONS=true

failed=()
step() {
  echo "==> $*"
  "$@" || failed+=("$*")
}

# Build the wheel and run it from outside the checkout, so a file the package needs but
# the wheel leaves out fails here instead of on a user's git install.
package_smoke() {
  local out
  out="$(mktemp -d)" || return 1
  uv build -q --wheel --out-dir "$out" &&
    (cd "$out" && uv run -q --isolated --no-project --python 3.13 \
      --with "$out"/*.whl axial --help >/dev/null)
  local status=$?
  rm -rf "$out"
  return $status
}

step uv lock --check
step uv run ruff check .
step uv run mypy src/axial scripts
step uv run pytest -q
step env UV_PROJECT_ENVIRONMENT=.venv-3.13 uv run --python 3.13 pytest -q
step uv run python scripts/knowledge_index.py --check
step package_smoke

if ((${#failed[@]})); then
  echo "==> failed:"
  printf '    %s\n' "${failed[@]}"
  # A forced run that fails takes back an earlier pass on this tree.
  [[ -n "$tree" ]] && rm -f "$stamps/$tree"
  remind_about_the_hook
  exit 1
fi
echo "==> all checks passed"

# The hash is taken again because a file edited during the run may not be what the checks saw.
if [[ -z "$git_dir" ]]; then
  echo "==> git finds no repository here, so this pass is not recorded"
elif [[ -z "$tree" ]]; then
  echo "==> could not hash the working tree, so this pass is not recorded"
elif [[ "$(working_tree)" != "$tree" ]]; then
  echo "==> the working tree changed during the run, so this pass is not recorded"
elif ! { mkdir -p "$stamps" && touch "$stamps/$tree"; }; then
  echo "==> could not record this pass in $stamps, so the pre-push hook will not count it"
fi
remind_about_the_hook

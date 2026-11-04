"""Tests for scripts/check.sh's record of what passed and .githooks/pre-push, which reads it.

Each test runs the real scripts in a temporary clone, with a fake `uv` on PATH so that
check.sh's steps do nothing or fail when told to, and `git push` runs the real hook against
a local bare remote.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

FAKE_UV = """#!/bin/sh
echo "$*" >> "$UV_LOG"
if [ -n "${UV_EDITS:-}" ]; then echo edited >> "$UV_EDITS"; fi
exit "${UV_EXIT:-0}"
"""

REFUSAL = "have not passed scripts/check.sh"


@dataclass
class Clone:
    root: Path
    remote: Path
    env: dict[str, str]

    def run(
        self, *args: str | Path, cwd: Path | None = None, stdin: str | None = None, **env: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(arg) for arg in args],
            cwd=cwd or self.root,
            env={**self.env, **env},
            input=stdin,
            capture_output=True,
            text=True,
            check=False,
        )

    def git(self, *args: str | Path, cwd: Path | None = None) -> str:
        result = self.run("git", *args, cwd=cwd)
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    def write(self, name: str, text: str, cwd: Path | None = None) -> None:
        path = (cwd or self.root) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def commit(self, subject: str, cwd: Path | None = None) -> str:
        """Commit everything in the working tree and return the short hash."""
        self.git("add", "-A", cwd=cwd)
        self.git("commit", "-q", "--allow-empty", "-m", subject, cwd=cwd)
        return self.git("rev-parse", "--short", "HEAD", cwd=cwd)

    def check(self, *args: str, cwd: Path | None = None, **env: str) -> (
        subprocess.CompletedProcess[str]
    ):
        return self.run((cwd or self.root) / "scripts" / "check.sh", *args, cwd=cwd, **env)

    def push(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return self.run("git", "push", "-q", "origin", *args, cwd=cwd)

    def uv_runs(self) -> int:
        log = Path(self.env["UV_LOG"])
        return len(log.read_text().splitlines()) if log.exists() else 0

    def on_remote(self, ref: str = "main") -> str:
        return self.git("--git-dir", self.remote, "rev-parse", "--short", ref)


@pytest.fixture
def clone(tmp_path: Path) -> Clone:
    """A clone with the repo's two scripts committed and pushed, then the hook turned on."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(FAKE_UV)
    fake_uv.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env |= {
        "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "UV_LOG": str(tmp_path / "uv.log"),
    }
    remote = tmp_path / "remote.git"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", remote], env=env, check=True)
    subprocess.run(["git", "clone", "-q", remote, work], env=env, check=True, capture_output=True)

    made = Clone(work, remote, env)
    made.git("config", "user.name", "Test")
    made.git("config", "user.email", "test@example.com")
    (work / "scripts").mkdir()
    (work / ".githooks").mkdir()
    shutil.copy2(REPO / "scripts" / "check.sh", work / "scripts" / "check.sh")
    shutil.copy2(REPO / ".githooks" / "pre-push", work / ".githooks" / "pre-push")
    made.write(".gitignore", ".venv/\n")
    made.write("a.txt", "base\n")
    made.commit("base")
    made.git("push", "-q", "origin", "main")
    made.git("config", "core.hooksPath", ".githooks")
    return made


def assert_refused(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode != 0
    assert REFUSAL in result.stderr


# What a run of check.sh records.


def test_a_commit_of_the_tree_that_passed_can_be_pushed(clone: Clone) -> None:
    clone.write("a.txt", "changed\n")
    assert clone.check().returncode == 0
    commit = clone.commit("the change")

    assert clone.push("main").returncode == 0
    assert clone.on_remote() == commit


def test_a_failed_check_does_not_count(clone: Clone) -> None:
    clone.write("a.txt", "changed\n")
    assert clone.check(UV_EXIT="1").returncode == 1
    clone.commit("the change")

    assert_refused(clone.push("main"))


def test_a_run_changes_nothing_in_the_index_or_the_working_tree(clone: Clone) -> None:
    clone.write("a.txt", "staged\n")
    clone.git("add", "a.txt")
    clone.write("a.txt", "staged then edited\n")
    clone.write("b.txt", "untracked\n")
    before = clone.git("status", "--porcelain")

    assert clone.check().returncode == 0

    assert clone.git("status", "--porcelain") == before
    assert (clone.root / "a.txt").read_text() == "staged then edited\n"


def test_a_tracked_file_matching_gitignore_stays_in_the_tree_that_passed(clone: Clone) -> None:
    clone.write(".venv/kept.txt", "tracked on purpose\n")
    clone.git("add", "-f", ".venv/kept.txt")
    clone.git("commit", "-q", "-m", "track a file gitignore matches")
    assert clone.check().returncode == 0

    assert clone.push("main").returncode == 0


def test_an_ignored_file_is_not_part_of_the_tree_that_passed(clone: Clone) -> None:
    clone.write("a.txt", "changed\n")
    clone.write(".venv/noise.txt", "not in any commit\n")
    assert clone.check().returncode == 0
    clone.commit("the change")

    assert clone.push("main").returncode == 0


def test_an_untracked_file_is_part_of_the_tree_that_passed(clone: Clone) -> None:
    clone.write("a.txt", "changed\n")
    clone.write("scratch.txt", "the tests ran with this here\n")
    assert clone.check().returncode == 0
    clone.git("add", "a.txt")
    clone.git("commit", "-q", "-m", "the change without the scratch file")

    assert_refused(clone.push("main"))


def test_an_edit_only_the_index_timestamp_can_reveal_is_in_the_tree_that_passed(
    clone: Clone,
) -> None:
    """Same size and same mtime, in a file staged as the index was last written: git reads the
    file again only because the index is no older than the entry, so a copy of the index that
    takes a new timestamp would record the old contents."""
    clone.git("config", "core.trustctime", "false")
    long_ago = 1_500_000_000
    path = clone.root / "a.txt"
    path.write_text("aaaa\n")
    os.utime(path, (long_ago, long_ago))
    clone.git("add", "a.txt")
    os.utime(clone.root / ".git" / "index", (long_ago, long_ago))
    path.write_text("bbbb\n")
    os.utime(path, (long_ago, long_ago))

    assert clone.check().returncode == 0
    clone.commit("the edit")

    assert clone.push("main").returncode == 0


def test_a_file_edited_during_the_run_means_no_tree_is_recorded(clone: Clone) -> None:
    clone.write("a.txt", "as it began\n")
    result = clone.check(UV_EDITS=str(clone.root / "a.txt"))
    assert result.returncode == 0
    assert "changed during the run" in result.stdout

    clone.commit("as it ended")
    assert_refused(clone.push("main"))
    clone.write("a.txt", "as it began\n")
    clone.commit("as it began")
    assert_refused(clone.push("main"))


def test_a_second_run_on_the_same_tree_stops_early(clone: Clone) -> None:
    assert clone.check().returncode == 0
    runs = clone.uv_runs()

    again = clone.check()

    assert again.returncode == 0
    assert "already checked" in again.stdout
    assert clone.uv_runs() == runs


def test_force_runs_the_checks_on_a_tree_that_already_passed(clone: Clone) -> None:
    assert clone.check().returncode == 0
    runs = clone.uv_runs()

    forced = clone.check("--force", UV_EXIT="1")

    assert forced.returncode == 1
    assert clone.uv_runs() > runs


def test_a_failed_forced_run_takes_back_the_earlier_pass(clone: Clone) -> None:
    clone.write("a.txt", "changed\n")
    assert clone.check().returncode == 0
    assert clone.check("--force", UV_EXIT="1").returncode == 1
    clone.commit("the change")

    assert_refused(clone.push("main"))


def test_a_failed_run_keeps_the_passes_of_other_trees(clone: Clone) -> None:
    assert clone.check().returncode == 0
    clone.write("a.txt", "changed\n")
    assert clone.check(UV_EXIT="1").returncode == 1
    clone.write("a.txt", "base\n")

    assert "already checked" in clone.check().stdout


def test_a_tree_that_cannot_be_hashed_runs_and_says_the_pass_is_not_recorded(clone: Clone) -> None:
    assert clone.check().returncode == 0
    runs = clone.uv_runs()
    assert clone.run("git", "init", "-q", clone.root / "nested").returncode == 0

    result = clone.check()

    assert result.returncode == 0
    assert clone.uv_runs() > runs
    assert "could not hash the working tree, so this pass is not recorded" in result.stdout
    assert "does not have a commit checked out" in result.stderr


def test_a_nested_repository_with_a_commit_says_the_pass_is_not_recorded(clone: Clone) -> None:
    """Git would hash it as a link to its commit, a tree no commit here can have, so check.sh
    would say "already checked" while the hook refused, with neither saying why."""
    nested = clone.root / "nested"
    assert clone.run("git", "init", "-q", nested).returncode == 0
    clone.git("config", "user.name", "Test", cwd=nested)
    clone.git("config", "user.email", "test@example.com", cwd=nested)
    clone.commit("inside", cwd=nested)

    result = clone.check()

    assert result.returncode == 0
    assert "could not hash the working tree, so this pass is not recorded" in result.stdout
    assert "a git repository sits inside this one" in result.stderr
    assert "hint:" not in result.stderr
    assert "already checked" not in clone.check().stdout


def add_a_submodule(clone: Clone) -> Path:
    """Commit a nested repository's commit as a link in the index, as a real submodule is."""
    nested = clone.root / "a submodule"
    assert clone.run("git", "init", "-q", nested).returncode == 0
    clone.git("config", "user.name", "Test", cwd=nested)
    clone.git("config", "user.email", "test@example.com", cwd=nested)
    clone.commit("inside", cwd=nested)
    clone.commit("add a submodule")
    return nested


def test_a_real_submodule_in_the_index_can_pass_and_be_pushed(clone: Clone) -> None:
    add_a_submodule(clone)

    result = clone.check()

    assert result.returncode == 0
    assert "not recorded" not in result.stdout
    assert "already checked" in clone.check().stdout
    assert clone.push("main").returncode == 0


def test_a_deleted_submodule_folder_is_not_called_a_nested_repository(clone: Clone) -> None:
    shutil.rmtree(add_a_submodule(clone))

    result = clone.check()

    assert result.returncode == 0
    assert "sits inside" not in result.stderr
    assert "not recorded" not in result.stdout


def test_a_run_leaves_no_temporary_files(clone: Clone, tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    assert clone.check(TMPDIR=str(scratch)).returncode == 0

    assert list(scratch.iterdir()) == []


def test_a_pass_that_cannot_be_written_says_so(clone: Clone) -> None:
    (clone.root / ".git" / "axial-checked").write_text("a file where the directory goes\n")

    result = clone.check()

    assert result.returncode == 0
    assert "could not record this pass" in result.stdout


def test_a_changed_tree_runs_again(clone: Clone) -> None:
    assert clone.check().returncode == 0
    runs = clone.uv_runs()
    clone.write("a.txt", "changed\n")

    assert "already checked" not in clone.check().stdout
    assert clone.uv_runs() > runs


def test_an_unknown_argument_is_refused_before_any_check_runs(clone: Clone) -> None:
    result = clone.check("--bogus")

    assert result.returncode == 2
    assert "usage: scripts/check.sh [--force]" in result.stderr
    assert clone.uv_runs() == 0


def test_a_clone_with_the_hook_off_is_told_how_to_turn_it_on(clone: Clone) -> None:
    turn_on = "git config core.hooksPath .githooks"
    assert turn_on not in clone.check().stdout

    clone.git("config", "--unset", "core.hooksPath")

    assert turn_on in clone.check().stdout  # the run that stops early
    assert turn_on in clone.check("--force").stdout
    assert turn_on in clone.check("--force", UV_EXIT="1").stdout


@pytest.mark.parametrize("spelling", ["./.githooks", ".githooks/", "absolute"])
def test_the_hook_counts_as_on_however_its_path_is_written(clone: Clone, spelling: str) -> None:
    written = str(clone.root / ".githooks") if spelling == "absolute" else spelling
    clone.git("config", "core.hooksPath", written)

    assert "pre-push hook is off" not in clone.check().stdout
    clone.write("a.txt", "changed\n")
    clone.commit("unchecked")
    assert_refused(clone.push("main"))


def test_a_copy_of_the_hook_in_the_default_place_counts_as_on(clone: Clone) -> None:
    clone.git("config", "--unset", "core.hooksPath")
    hooks = clone.root / ".git" / "hooks"
    hooks.mkdir(exist_ok=True)
    shutil.copy2(clone.root / ".githooks" / "pre-push", hooks / "pre-push")

    assert "pre-push hook is off" not in clone.check().stdout


def test_a_copy_git_would_not_run_does_not_count_as_on(clone: Clone) -> None:
    clone.git("config", "--unset", "core.hooksPath")
    hooks = clone.root / ".git" / "hooks"
    hooks.mkdir(exist_ok=True)
    shutil.copyfile(clone.root / ".githooks" / "pre-push", hooks / "pre-push")
    (hooks / "pre-push").chmod(0o644)

    assert "pre-push hook is off" in clone.check().stdout


def test_another_hook_in_the_default_place_does_not_count_as_on(clone: Clone) -> None:
    clone.git("config", "--unset", "core.hooksPath")
    hooks = clone.root / ".git" / "hooks"
    hooks.mkdir(exist_ok=True)
    (hooks / "pre-push").write_text("#!/bin/sh\nexit 0\n")
    (hooks / "pre-push").chmod(0o755)

    assert "pre-push hook is off" in clone.check().stdout


def test_outside_a_git_checkout_the_checks_still_run(clone: Clone, tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    (plain / "scripts").mkdir(parents=True)
    shutil.copy2(clone.root / "scripts" / "check.sh", plain / "scripts" / "check.sh")

    result = clone.run(plain / "scripts" / "check.sh", cwd=plain)

    assert result.returncode == 0
    assert "all checks passed" in result.stdout
    assert clone.uv_runs() > 0
    assert "git finds no repository here, so this pass is not recorded" in result.stdout
    assert "pre-push hook" not in result.stdout
    assert "fatal" not in result.stderr


# Which commits the hook holds to it.


def test_a_commit_that_never_passed_is_refused_by_name(clone: Clone) -> None:
    before = clone.on_remote()
    commit = clone.commit("no check for this one")

    result = clone.push("main")

    assert_refused(result)
    assert f"{commit} no check for this one" in result.stderr
    assert clone.on_remote() == before


def test_every_new_commit_is_held_to_it_and_not_only_the_last(clone: Clone) -> None:
    clone.write("a.txt", "first\n")
    first = clone.commit("first")
    clone.write("a.txt", "second\n")
    assert clone.check().returncode == 0
    second = clone.commit("second")

    result = clone.push("main")

    assert_refused(result)
    assert f"{first} first" in result.stderr
    assert second not in result.stderr


def test_following_the_refusal_for_an_earlier_commit_gets_it_pushed(clone: Clone) -> None:
    clone.write("a.txt", "first\n")
    first = clone.commit("first")
    clone.write("a.txt", "second\n")
    assert clone.check().returncode == 0
    clone.commit("second")

    refusal = clone.push("main")
    assert "git switch --detach <commit> && scripts/check.sh" in refusal.stderr

    clone.git("switch", "-q", "--detach", first)
    assert clone.check().returncode == 0
    clone.git("switch", "-q", "main")

    assert clone.push("main").returncode == 0


def test_a_commit_the_hook_cannot_list_refuses_the_push(clone: Clone) -> None:
    unknown = "1" * 40
    zero = "0" * 40

    result = clone.run(
        clone.root / ".githooks" / "pre-push",
        "origin",
        clone.remote,
        stdin=f"refs/heads/main {unknown} refs/heads/main {zero}\n",
    )

    assert result.returncode == 1
    assert f"could not list the commits in {unknown}" in result.stderr


def test_a_merge_commit_needs_its_own_pass(clone: Clone, tmp_path: Path) -> None:
    other = tmp_path / "other"
    clone.git("clone", "-q", clone.remote, other)
    clone.git("config", "user.name", "Test", cwd=other)
    clone.git("config", "user.email", "test@example.com", cwd=other)
    clone.write("theirs.txt", "from someone else\n", cwd=other)
    theirs = clone.commit("theirs", cwd=other)
    clone.git("push", "-q", "--no-verify", "origin", "main", cwd=other)

    clone.write("ours.txt", "from here\n")
    assert clone.check().returncode == 0
    ours = clone.commit("ours")
    clone.git("fetch", "-q", "origin")
    clone.git("merge", "-q", "--no-edit", "origin/main")
    merge = clone.git("rev-parse", "--short", "HEAD")

    result = clone.push("main")

    assert_refused(result)
    assert merge in result.stderr
    assert ours not in result.stderr
    assert theirs not in result.stderr

    assert clone.check().returncode == 0
    assert clone.push("main").returncode == 0
    assert clone.on_remote() == merge


def test_a_new_branch_is_held_to_it_for_its_own_commits_only(clone: Clone) -> None:
    clone.git("switch", "-q", "-c", "feature")
    clone.write("a.txt", "on the branch\n")
    assert clone.commit("on the branch")

    result = clone.push("feature")
    assert_refused(result)
    assert "base" not in result.stderr

    assert clone.check().returncode == 0
    assert clone.push("feature").returncode == 0


def test_a_branch_on_top_of_a_pushed_branch_skips_the_commits_already_pushed(clone: Clone) -> None:
    clone.git("switch", "-q", "-c", "lower")
    clone.write("a.txt", "lower\n")
    clone.commit("lower")
    clone.git("push", "-q", "--no-verify", "origin", "lower")

    clone.git("switch", "-q", "-c", "upper")
    clone.write("a.txt", "upper\n")
    assert clone.check().returncode == 0
    clone.commit("upper")

    assert clone.push("upper").returncode == 0


def test_a_force_push_over_a_commit_this_clone_never_saw(clone: Clone, tmp_path: Path) -> None:
    other = tmp_path / "other"
    clone.git("clone", "-q", clone.remote, other)
    clone.git("config", "user.name", "Test", cwd=other)
    clone.git("config", "user.email", "test@example.com", cwd=other)
    clone.write("theirs.txt", "unseen here\n", cwd=other)
    clone.commit("theirs", cwd=other)
    clone.git("push", "-q", "--no-verify", "origin", "main", cwd=other)

    clone.write("a.txt", "ours\n")
    unchecked = clone.commit("ours")
    result = clone.push("--force", "main")
    assert_refused(result)
    assert f"{unchecked} ours" in result.stderr

    assert clone.check().returncode == 0
    assert clone.push("--force", "main").returncode == 0
    assert clone.on_remote() == unchecked


def test_what_the_target_branch_holds_is_left_alone_with_no_tracking_ref(clone: Clone) -> None:
    clone.write("a.txt", "first\n")
    assert clone.check().returncode == 0
    clone.commit("first")
    assert clone.push("main").returncode == 0
    clone.git("update-ref", "-d", "refs/remotes/origin/main")

    clone.write("a.txt", "second\n")
    assert clone.check().returncode == 0
    clone.commit("second")

    assert clone.push("main").returncode == 0


def test_a_rewritten_commit_counts_as_new(clone: Clone) -> None:
    clone.write("a.txt", "first version\n")
    assert clone.check().returncode == 0
    clone.commit("the change")
    assert clone.push("main").returncode == 0

    clone.write("a.txt", "second version\n")
    clone.git("add", "-A")
    clone.git("commit", "-q", "--amend", "--no-edit")

    assert_refused(clone.push("--force", "main"))


def test_a_new_message_on_the_tree_that_passed_still_counts(clone: Clone) -> None:
    clone.write("a.txt", "changed\n")
    assert clone.check().returncode == 0
    clone.commit("first wording")
    clone.git("commit", "-q", "--amend", "-m", "second wording")

    assert clone.push("main").returncode == 0


def test_an_annotated_tag_is_held_to_it_for_the_commits_it_brings(clone: Clone) -> None:
    commit = clone.commit("tagged without a check")
    clone.git("tag", "-a", "v1", "-m", "v1")

    result = clone.push("v1")

    assert_refused(result)
    assert f"{commit} tagged without a check" in result.stderr


def test_a_tag_on_a_pushed_commit_is_left_alone(clone: Clone) -> None:
    clone.git("tag", "v0")

    assert clone.push("v0").returncode == 0


def test_a_push_to_any_other_ref_is_held_to_it(clone: Clone) -> None:
    commit = clone.commit("for review")

    result = clone.push("HEAD:refs/for/main")

    assert_refused(result)
    assert f"{commit} for review" in result.stderr


def test_a_notes_ref_is_left_alone(clone: Clone) -> None:
    """A notes commit never goes through check.sh, and it changes no code."""
    clone.git("notes", "add", "-m", "a note", "HEAD")

    assert clone.push("refs/notes/commits").returncode == 0


def test_deleting_a_remote_branch_is_allowed(clone: Clone) -> None:
    clone.git("switch", "-q", "-c", "feature")
    clone.commit("to be deleted")
    clone.git("push", "-q", "--no-verify", "origin", "feature")

    assert clone.push(":feature").returncode == 0
    assert clone.run("git", "--git-dir", clone.remote, "rev-parse", "-q", "--verify",
                     "refs/heads/feature").returncode != 0


def test_a_commit_two_refs_share_is_named_once(clone: Clone) -> None:
    clone.git("switch", "-q", "-c", "feature")
    clone.commit("on both refs")
    clone.git("branch", "feature-copy")

    result = clone.push("feature", "feature-copy")

    assert_refused(result)
    assert result.stderr.count("on both refs") == 1


def test_a_pass_in_one_worktree_counts_in_the_others(clone: Clone, tmp_path: Path) -> None:
    linked = tmp_path / "linked"
    clone.git("worktree", "add", "-q", "-b", "feature", linked)
    clone.write("a.txt", "in the linked worktree\n", cwd=linked)
    assert clone.check(cwd=linked).returncode == 0
    clone.commit("in the linked worktree", cwd=linked)

    assert clone.push("feature").returncode == 0

    clone.write("a.txt", "and again\n", cwd=linked)
    assert clone.check(cwd=linked).returncode == 0
    clone.commit("and again", cwd=linked)

    assert clone.push("feature", cwd=linked).returncode == 0


# What it does not catch, and how to get past it.


def test_a_branch_on_another_remote_exempts_its_commits(clone: Clone, tmp_path: Path) -> None:
    clone.git("init", "-q", "--bare", tmp_path / "second.git")
    clone.git("remote", "add", "second", tmp_path / "second.git")
    clone.commit("pushed to the second remote without a check")
    clone.git("push", "-q", "--no-verify", "second", "main")

    assert clone.push("main").returncode == 0


def test_a_branch_without_the_hook_file_runs_no_hook(clone: Clone) -> None:
    clone.git("rm", "-q", ".githooks/pre-push")
    clone.commit("drop the hook")

    assert clone.push("main").returncode == 0


def test_a_clone_with_no_tracking_refs_holds_its_whole_history_to_it(clone: Clone) -> None:
    clone.git("update-ref", "-d", "refs/remotes/origin/main")
    clone.git("switch", "-q", "-c", "feature")

    result = clone.push("feature")

    assert_refused(result)
    assert "base" in result.stderr


def test_a_commit_on_a_check_from_before_the_record_gets_through_after_a_rebase(
    clone: Clone,
) -> None:
    clone.git("switch", "-q", "-c", "old-base")
    clone.write("scripts/check.sh", "#!/bin/sh\nexit 0\n")
    clone.commit("a check.sh that records nothing")
    clone.git("push", "-q", "--no-verify", "origin", "old-base")
    clone.git("switch", "-q", "-c", "feature")
    clone.write("b.txt", "the feature\n")
    clone.commit("the feature")

    assert clone.check().returncode == 0
    assert_refused(clone.push("feature"))

    clone.git("rebase", "-q", "--onto", "main", "old-base", "feature")

    assert clone.check().returncode == 0
    assert clone.push("feature").returncode == 0

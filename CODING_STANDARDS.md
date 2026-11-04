# Coding standards

The review step reads this file. Each rule here needs judgement; a rule a tool can check belongs in ruff, mypy or a test.

## Every behaviour a commit claims has a test that fails without it

A commit message or PR description that says behaviour changed (a fix, an input now accepted, a new refusal) is backed by a test that fails when the change is reverted. Check claims one by one: a message saying five readers now accept something needs a test for each reader, or it names the one it tested.

A test that already passes before the change guards against a looser rewrite. It is welcome, and the commit message names it as a guard.

To check a commit, run `uv run python scripts/revert_check.py <commit>`. In a temporary checkout of the commit, it reverses the source changes and runs the tests the commit added or changed. Every one of those tests fails while the change is reversed, apart from the named guards, each of which you pass with its own `--guard <test id>` as well as naming it in the commit message. For a test that arrived in a later commit, name the test and add `--at <that commit>`. A test that fails only on a missing import or module attribute usually shows that a name the commit adds exists and nothing more, so the script exits 1 and the review reads what the test asserts. The script also prints what every other test failed on, and the review reads that too: a new keyword or method fails there just as a real bug would, and shows no more.

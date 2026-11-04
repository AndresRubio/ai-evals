# Coding standards

The review step reads this file. Each rule here needs judgement; a rule a tool can check belongs in ruff, mypy or a test.

## Every behaviour a commit claims has a test

A commit message or PR description that says behaviour changed (a fix, an input now accepted, a new refusal) is backed by a test that fails when the change is reverted. Check claims one by one: a message saying five readers now accept something needs a test for each reader.

A claim the reader will act on needs a test even when the behaviour predates the commit: how to recover from a new refusal, which commands keep working after it, or a limit they would plan around. Saying a claim is untested is enough only for a remark nobody will rely on. A fact about a system outside the repo, such as what CI runs, cannot have a test here, so the message says where it comes from.

A test that already passes before the change guards against a looser rewrite. It is welcome, and the commit message names it as a guard.

To check a commit, run `uv run python scripts/revert_check.py <commit>`. In a temporary checkout of the commit, it reverses the source changes and runs the tests the commit added or changed. Every one of those tests fails while the change is reversed, apart from the named guards, each of which you pass with its own `--guard <test id>` as well as naming it in the commit message. For a test that arrived in a later commit, name the test and add `--at <that commit>`. A test that fails only on a missing import or module attribute usually shows that a name the commit adds exists and nothing more, so the script exits 1 and the review reads what the test asserts. The script also prints what every other test failed on, and the review reads that too: a new keyword or method fails there just as a real bug would, and shows no more.

## A new refusal has a test that the nearest valid input still passes

A refusal that reaches too far stops someone's valid input, and a test of the refused input cannot see that. So a new refusal comes with a test on the valid input closest to it: a whole emoji (an escaped surrogate pair) for a refusal of a lone surrogate, or a 100-character name for a 100-character limit. If that test passes with the change reversed, as it usually will, the commit message names it as a guard.

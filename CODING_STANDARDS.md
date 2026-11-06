# Coding standards

The review step reads this file. Each rule here needs judgement; a rule a tool can check belongs in ruff, mypy or a test.

## Every behaviour a commit claims has a test

A commit message or PR description that says behaviour changed (a fix, an input now accepted, a new refusal) is backed by a test that fails when the change is reverted. Check claims one by one: a message saying five readers now accept something needs a test for each reader.

A claim the reader will act on needs a test even when the behaviour predates the commit: how to recover from a new refusal, which commands keep working after it, or a limit they would plan around. Saying a claim is untested is enough only for a remark nobody will rely on. A fact about a system outside the repo, such as what CI runs, cannot have a test here, so the message says where it comes from.

A test that already passes before the change guards against a looser rewrite. It is welcome, and the commit message names it as a guard.

To check a commit, run `uv run python scripts/revert_check.py <commit>`. In a temporary checkout of the commit, it reverses the source changes and runs the tests the commit added or changed. Every one of those tests fails while the change is reversed, apart from the named guards, each of which you pass with its own `--guard <test id>` as well as naming it in the commit message. For a test that arrived in a later commit, name the test and add `--at <that commit>`. A test that fails only on a missing import or module attribute usually shows that a name the commit adds exists and nothing more, so the script exits 1 and the review reads what the test asserts. The script also prints what every other test failed on, and the review reads that too: a new keyword or method fails there just as a real bug would, and shows no more.

## A new refusal has a test that the nearest valid input still passes

A refusal that reaches too far stops someone's valid input, and a test of the refused input cannot see that. So a new refusal comes with a test on the valid input closest to it: a whole emoji (an escaped surrogate pair) for a refusal of a lone surrogate, or a 100-character name for a 100-character limit. If that test passes with the change reversed, as it usually will, the commit message names it as a guard.

## A commit that says nothing changed is read as if it did

A move, a hoist or a rename comes with the claim that behaviour stayed the same, and that claim gets no test, so the review checks it line by line: every condition, default and exception clause in the old code has its counterpart in the new, and a difference is either a behaviour change the message states and a test covers, or a defect. A draft of the judge hoist carried two such changes under a message saying nothing changed: a mode missing from the taxonomy lost the mode id it used to get as its name, and a failure to write judge.yaml stopped reaching the user.

For a move, start from `uv run python scripts/moved_code.py <base> src tests`. It files every top-level statement under the name it defines, together with the comments above it and its docstring, and prints each name whose text differs between the base and the working tree, whichever file holds it now. It also prints a statement that moved with its text intact when the new module may give it another meaning, for example because it reads `__file__`, has a relative import, or uses a name that is another statement in the new module. Pass a rename the commit makes on purpose as `--rename old=new`, and add `--diff` to see the lines. Each name it prints is a change the message states or a defect, and a reworded comment or docstring counts. The package move was first described as moving code unchanged while it reworded three module docstrings and one comment.

The script is where the reading starts. It leaves out top-level imports, so it cannot tell when a name the moved code imports is another thing in the new module. It does not compare the order of statements, and it reads Python files only, so a moved template or page is read by hand. `--rename` also renames a parameter or a local that shares the name, and the count it prints is the check on that.

Tests move with the code they test. A test of a function that moved to another module lives in that module's test file, so a reader looking for the function's tests finds them beside it, and the old file stops carrying tests of code its module no longer holds.

## A change to a set of rules says what it newly allows and what it newly refuses

Import contracts and lint settings are rules about other code. A rewrite of them can pass every test and still let through what the old set refused, because the tests that pass were written for the new set. So the message lists the differences in both directions: what the new rules allow that the old ones refused, and what they refuse that the old ones allowed. Each entry has a test, or a reason it is wanted. The first draft of the package layout lost two protections without saying so: the statistics could import the model client, and pairing could import the alignment report.

The tests of the import contracts are checked by changing the contracts. `uv run python scripts/weaken_rules.py` makes one change at a time in a copy of the repository, such as swapping two rows or leaving a contract with nothing to refuse, and runs `tests/test_layers.py` against the copy. A change the tests fail on is caught. For a change that survives, a test is added, or the message names the change and says why no test can tell the two sets of rules apart.

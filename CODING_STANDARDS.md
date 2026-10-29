# Coding standards

The review step reads this file. Each rule here needs judgement; a rule a tool can check belongs in ruff, mypy or a test.

## Every behaviour a commit claims has a test that fails without it

A commit message or PR description that says behaviour changed (a fix, an input now accepted, a new refusal) is backed by a test that fails when the change is reverted. Check claims one by one: a message saying five readers now accept something needs a test for each reader, or it names the one it tested.

A test that already passes before the change guards against a looser rewrite. It is welcome, and the commit message names it as a guard.

To check a commit: reverse its source changes with `git show <commit> -- src | git apply -R`, run the tests the commit added, then restore with `git checkout HEAD -- src`. Every added test fails while the change is reversed, apart from the named guards. If the reverse patch does not apply because a later commit edited the same lines, run the check from a checkout of the commit itself.

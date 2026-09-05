## Description and Motivation

<!--

    Please write a description of what this PR is changing, removing or adding, and why.
    Consider including before/after comparisons.

    For this recipe, a good description usually covers:
      * which `.env` knob, default, script behavior, or runtime patch changes
      * whether the change affects measured numbers (KV cache budget, TTFT, decode
        throughput, fairness spread, concurrency) and, if so, the expected direction
      * why the change is safe on both ranks of the two-node TP=2 lane
      * if it adds a runtime patch: why it follows the repo's pattern (opt-in
        `DSPARK_ENABLE_*` flag defaulting to 0, source-exact/fail-closed patcher
        with fixtures and a --status/--check path)

-->

## Related Issues

<!--

    Add the list of issues related to this PR from the issue tracker.
    Indicate which of these issues are resolved or fixed by this PR, like #XXXX, where XXXX is the issue number.

-->

---

## Testing

<!--

    Tell us how you verified this change. For this recipe that usually means:

      * `bash scripts/ci-validate.sh` (CPU recipe gates, required)
      * the relevant focused suites, e.g. scripts/test-*.py / tests/test_*.py
      * for shell changes: `bash -n` on every touched script
      * for runtime/patch changes: a boot on the two-node lane with the flag both
        off and on, and the measured numbers (see README "What speed to expect"
        and docs/ENVS.md for the format used)
      * if the change touches a hotfix: the fixture/identity tests pass and the
        patcher refuses drifted bytes fail-closed

-->

---

## Checklist:

<!--

    Thanks for contributing to Mia's AI Lab!

    Before you file this pull request, please follow the items on this checklist and
    put an x in each of the boxes, like this: [x].

-->

- [ ] I have read the README and `docs/ENVS.md` and kept my changes consistent with them.
- [ ] My pull request has a sound title and description (not something vague like `Update README.md`).
- [ ] My change is reproducible and verified (CI validate, focused suites, and — for runtime changes — a measured boot).
- [ ] New runtime behavior is opt-in and default-off, matching the recipe's existing knobs; defaults are unchanged unless the PR is explicitly about a default.
- [ ] I updated the README, `docs/ENVS.md`, `.env.dspark.example`, and `CHANGELOG.md` for any knob, default, or behavior I changed.
- [ ] `.env.dspark.example` and the Compose fallback agree for every knob I touched (they are locked together in `scripts/ci-validate.sh`).

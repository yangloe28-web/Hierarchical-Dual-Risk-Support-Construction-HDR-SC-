# Publication checklist

Review these items before publishing or updating the public repository.

- [ ] Create a new repository with fresh Git history. Do not copy the current
      research repository's `.git` directory.
- [ ] Confirm institutional intellectual-property requirements.
- [ ] If an open-source release is intended, obtain approval and replace
      `LICENSE-PENDING.md` with an approved `LICENSE` file. Otherwise, clearly
      describe the repository as publicly visible source without a license.
- [ ] Confirm that no third-party detector source has been copied into this
      repository.
- [ ] Pin or document the external detector versions used by each adapter.
- [ ] Verify every optional detector environment on a clean machine before
      claiming end-to-end reproducibility.
- [ ] Run `python tools/run_tests.py` successfully.
- [ ] Run `python tools/audit_release.py` successfully.
- [ ] Inspect the complete Git diff before the first public push.
- [ ] Confirm that no datasets, model weights, caches, results, manuscripts,
      figures, spreadsheets, local paths, credentials, or private metadata are
      tracked.
- [ ] State accurately whether the repository is a method implementation or a
      complete experiment-reproduction package.

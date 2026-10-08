# MLflow skills sync

`sync-mlflow-skills.yml` reacts to every push to this repository's `main` branch,
including merges, and opens or refreshes a PR against `mlflow/mlflow:master`.
It resolves the latest `main` commit when the publishing job runs, so queued
events are coalesced into one up-to-date PR rather than one PR per commit.

The updater changes only `mlflow/assistant/skills`. It uses GitHub's Git database
API to create the gitlink tree and a DCO-signed-off commit; it does not run skill
code or check out MLflow with a write token. Updates retain branch history and
use fast-forward-only ref updates. Human edits to the reserved
`automation/update-mlflow-skills` branch cause the workflow to fail rather than
being overwritten.

PRs require normal MLflow CI and review. The workflow does not enable auto-merge
or update release branches. If MLflow already pins the latest skills, it does
nothing, or closes its obsolete open PR if the update landed separately.

## One-time setup

Reuse the GitHub App used by MLflow's existing automated PR workflows:

1. Ensure the app is installed on `mlflow/mlflow` with **Contents: write** and
   **Pull requests: write**. No Workflows or organization-wide write permission
   is needed for a submodule-only change.
2. Make the following Actions secrets available to `mlflow/skills`, either as
   repository secrets or organization secrets restricted to this repository:
   - `APP_CLIENT_ID`: the existing app's client ID.
   - `APP_PRIVATE_KEY`: the app's private key.
3. Merge the workflow into `main`. The push triggers the first sync. Use
   **Actions → Sync skills to MLflow → Run workflow**, selecting `main`, to retry
   after configuring credentials or resolving a failed run.

Only publishing jobs on this repository's `main` branch receive the app token.
PR runs and forks run unit tests only. The app token is explicitly scoped to
`mlflow/mlflow`, while the default read-only token resolves the source commit.
The default `GITHUB_TOKEN` alone cannot write to another repository.

DCO sign-off is a commit-message trailer, not a cryptographic signature. Any
rules requiring cryptographically signed commits on the automation branch need
to allow this app's Git database API commits.

## Tests

No package installation or live GitHub credentials are required:

```bash
node --test .github/scripts/sync-mlflow-skills.test.js
```

The tests use a fake REST client to cover PR creation and reuse, repeat runs,
safe branch updates, no-op/superseded updates, and failure handling.

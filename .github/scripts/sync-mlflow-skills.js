const TARGET = { owner: "mlflow", repo: "mlflow" };
const BASE = "master";
const BRANCH = "automation/update-mlflow-skills";
const PATH = "mlflow/assistant/skills";
const SOURCE_URL = "https://github.com/mlflow/skills";
const TITLE = "Update bundled `mlflow/assistant/skills`";
const COMMIT_TITLE = "Update bundled MLflow skills";

function prBody(previousSha, sourceSha) {
  return `### Related Issues/PRs

Relates to ${SOURCE_URL}/compare/${previousSha}...${sourceSha}.

### What changes are proposed in this pull request?

Update \`${PATH}\` from \`${previousSha}\` to \`${sourceSha}\`, the latest commit on \`mlflow/skills:main\`.

Only the submodule pointer changes. This PR is refreshed when upstream skills advance; normal review and CI are required, and auto-merge is not enabled.

### How is this PR tested?

- [x] Existing unit/integration tests

The existing Assistant installer tests should run in MLflow CI. This workflow does not execute MLflow tests before opening the PR. The updater has its own unit tests and checks that an existing bot branch contains only a submodule-pointer change before updating it.

### Does this PR require documentation update?

- [x] No.

### Does this PR require updating the [MLflow Skills](${SOURCE_URL}) repository?

- [x] No.

This PR consumes changes already merged into the skills repository.

### Release Notes

#### Is this a user-facing change?

- [x] Yes. Refresh the MLflow Assistant's bundled skills with the latest upstream guidance.

#### What component(s), interfaces, languages, and integrations does this PR affect?

Components

- [x] \`area/docs\`: MLflow documentation pages

<a name="release-note-category"></a>

#### How should the PR be classified in the release notes? Choose one:

- [x] \`rn/documentation\` - A user-facing documentation change worth mentioning in the release notes

#### Is this PR a critical bugfix or security fix that should go into the next patch release?

- [ ] This PR is critical and needs to be in the next patch release
- [x] This PR can wait for the next minor release
`;
}

module.exports = async function sync({ github, core, sourceSha, appSlug }) {
  if (!/^[0-9a-f]{40}$/.test(sourceSha)) {
    throw new Error("Expected a full skills commit SHA");
  }
  if (!appSlug) {
    throw new Error("Expected the GitHub App slug");
  }

  async function getPin(ref) {
    const { data } = await github.rest.repos.getContent({
      ...TARGET,
      path: PATH,
      ref,
    });
    if (
      Array.isArray(data) ||
      data.submodule_git_url?.replace(/\.git$/, "") !== SOURCE_URL ||
      !/^[0-9a-f]{40}$/.test(data.sha)
    ) {
      throw new Error(`${PATH} must remain a submodule of ${SOURCE_URL}`);
    }
    return data.sha;
  }

  const { data: baseRef } = await github.rest.git.getRef({
    ...TARGET,
    ref: `heads/${BASE}`,
  });
  const baseSha = baseRef.object.sha;
  const previousSha = await getPin(baseSha);
  const prs = await github.paginate(github.rest.pulls.list, {
    ...TARGET,
    state: "open",
    base: BASE,
    head: `${TARGET.owner}:${BRANCH}`,
    per_page: 100,
  });
  const existingPr = prs[0];

  if (previousSha === sourceSha && !existingPr) {
    core.info("MLflow already bundles the latest skills");
    return { status: "up-to-date" };
  }

  let headSha;
  try {
    const { data } = await github.rest.git.getRef({
      ...TARGET,
      ref: `heads/${BRANCH}`,
    });
    headSha = data.object.sha;
  } catch (error) {
    if (error.status !== 404) throw error;
  }
  if (existingPr && !headSha) {
    throw new Error("The open skills PR is missing its bot branch");
  }

  const username = `${appSlug}[bot]`;
  const { data: bot } = await github.rest.users.getByUsername({ username });
  const author = {
    name: username,
    email: `${bot.id}+${username}@users.noreply.github.com`,
  };
  const signoff = `Signed-off-by: ${author.name} <${author.email}>`;
  let headCommit;
  let headPin;
  if (headSha) {
    const { data } = await github.rest.git.getCommit({
      ...TARGET,
      commit_sha: headSha,
    });
    headCommit = data;
    if (
      headCommit.author.name !== username ||
      !headCommit.message.startsWith(`${COMMIT_TITLE}\n\n`) ||
      !headCommit.message.includes(signoff) ||
      !headCommit.parents.length
    ) {
      throw new Error(
        `Refusing to overwrite an unrecognized branch: ${BRANCH}`,
      );
    }
    const { data: diff } = await github.rest.repos.compareCommits({
      ...TARGET,
      base: headCommit.parents[0].sha,
      head: headSha,
    });
    if (diff.files?.length !== 1 || diff.files[0].filename !== PATH) {
      throw new Error(`Refusing to overwrite human changes on ${BRANCH}`);
    }
    headPin = await getPin(headSha);
  }

  if (previousSha === sourceSha) {
    await github.rest.pulls.update({
      ...TARGET,
      pull_number: existingPr.number,
      state: "closed",
    });
    core.info(`Closed superseded skills PR: ${existingPr.html_url}`);
    return { status: "closed", url: existingPr.html_url };
  }

  const headIsCurrent =
    headCommit !== undefined &&
    headPin === sourceSha &&
    headCommit.parents.some(({ sha }) => sha === baseSha);
  if (headIsCurrent && existingPr) {
    core.info(`Skills PR is already up to date: ${existingPr.html_url}`);
    return { status: "up-to-date", url: existingPr.html_url, sha: headSha };
  }

  let commitSha = headSha;
  if (!headIsCurrent) {
    const { data: baseCommit } = await github.rest.git.getCommit({
      ...TARGET,
      commit_sha: baseSha,
    });
    const { data: tree } = await github.rest.git.createTree({
      ...TARGET,
      base_tree: baseCommit.tree.sha,
      tree: [{ path: PATH, mode: "160000", type: "commit", sha: sourceSha }],
    });
    // Retain the previous bot head as a parent so concurrent branch edits reject a
    // non-fast-forward update instead of being lost in a force-push.
    const parents = [...new Set([baseSha, headSha].filter(Boolean))];
    const { data: commit } = await github.rest.git.createCommit({
      ...TARGET,
      message: `${COMMIT_TITLE}\n\n${SOURCE_URL}/compare/${previousSha}...${sourceSha}\n\n${signoff}`,
      tree: tree.sha,
      parents,
      author,
      committer: author,
    });
    commitSha = commit.sha;
    if (headSha) {
      await github.rest.git.updateRef({
        ...TARGET,
        ref: `heads/${BRANCH}`,
        sha: commitSha,
        force: false,
      });
    } else {
      await github.rest.git.createRef({
        ...TARGET,
        ref: `refs/heads/${BRANCH}`,
        sha: commitSha,
      });
    }
  }

  if (existingPr) {
    await github.rest.pulls.update({
      ...TARGET,
      pull_number: existingPr.number,
      title: TITLE,
      body: prBody(previousSha, sourceSha),
    });
    core.info(`Refreshed skills PR: ${existingPr.html_url}`);
    return {
      status: headSha === commitSha ? "up-to-date" : "updated",
      url: existingPr.html_url,
      sha: commitSha,
    };
  }
  const { data: pr } = await github.rest.pulls.create({
    ...TARGET,
    base: BASE,
    head: BRANCH,
    title: TITLE,
    body: prBody(previousSha, sourceSha),
  });
  core.info(`Opened skills PR: ${pr.html_url}`);
  return { status: "created", url: pr.html_url, sha: commitSha };
};

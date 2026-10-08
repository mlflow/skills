const assert = require("node:assert/strict");
const test = require("node:test");

const sync = require("./sync-mlflow-skills");

const BASE = "a".repeat(40);
const SOURCE = "b".repeat(40);
const HEAD = "c".repeat(40);
const BASE_TREE = "d".repeat(40);
const CREATED_TREE = "e".repeat(40);
const CREATED_COMMIT = "f".repeat(40);
const OLD_SOURCE = "1".repeat(40);
const PREVIOUS_BASE = "2".repeat(40);
const PATH = "mlflow/assistant/skills";
const BRANCH = "automation/update-mlflow-skills";
const APP_SLUG = "mlflow-automation";
const BOT_NAME = `${APP_SLUG}[bot]`;
const BOT_ID = 12345;
const BOT_EMAIL = `${BOT_ID}+${BOT_NAME}@users.noreply.github.com`;
const PR_URL = "https://github.com/mlflow/mlflow/pull/42";

function apiError(status, message = `GitHub API error ${status}`) {
  return Object.assign(new Error(message), { status });
}

function fixture(options = {}) {
  const calls = [];
  const logs = [];
  const state = {
    head: options.existingHead ? HEAD : null,
    prs: options.existingPr ? [{ number: 42, html_url: PR_URL }] : [],
    sourceByCommit: {
      [BASE]: options.baseSource ?? OLD_SOURCE,
      [HEAD]: options.headSource ?? OLD_SOURCE,
    },
    commits: {
      [BASE]: {
        sha: BASE,
        tree: { sha: BASE_TREE },
        parents: [{ sha: PREVIOUS_BASE }],
      },
      [HEAD]: {
        sha: HEAD,
        tree: { sha: "3".repeat(40) },
        parents: [{ sha: options.headParent ?? PREVIOUS_BASE }],
        author: { name: BOT_NAME, email: BOT_EMAIL },
        message: `Update bundled MLflow skills\n\nSkills revision: ${OLD_SOURCE}\n\nSigned-off-by: ${BOT_NAME} <${BOT_EMAIL}>`,
        ...options.headCommit,
      },
    },
  };

  const record = (name, implementation) => async (params) => {
    calls.push({ name, params });
    if (!name.startsWith("users.")) {
      assert.equal(params.owner, "mlflow");
      assert.equal(params.repo, "mlflow");
    }
    const error = options.errors?.[name];
    if (error) throw error;
    return implementation(params);
  };

  const github = {
    rest: {
      git: {
        getRef: record("git.getRef", ({ ref }) => {
          if (ref === "heads/master")
            return { data: { object: { sha: BASE } } };
          assert.equal(ref, `heads/${BRANCH}`);
          if (options.headError) throw options.headError;
          if (!state.head) throw apiError(404);
          return { data: { object: { sha: state.head } } };
        }),
        getCommit: record("git.getCommit", ({ commit_sha }) => {
          assert.ok(state.commits[commit_sha], `Unknown commit: ${commit_sha}`);
          return { data: state.commits[commit_sha] };
        }),
        createTree: record("git.createTree", () => ({
          data: { sha: CREATED_TREE },
        })),
        createCommit: record("git.createCommit", (params) => {
          state.commits[CREATED_COMMIT] = {
            sha: CREATED_COMMIT,
            tree: { sha: params.tree },
            parents: params.parents.map((sha) => ({ sha })),
            author: params.author,
            message: params.message,
          };
          state.sourceByCommit[CREATED_COMMIT] = SOURCE;
          return { data: { sha: CREATED_COMMIT } };
        }),
        createRef: record("git.createRef", ({ ref, sha }) => {
          assert.equal(ref, `refs/heads/${BRANCH}`);
          state.head = sha;
          return { data: { object: { sha } } };
        }),
        updateRef: record("git.updateRef", ({ ref, sha }) => {
          assert.equal(ref, `heads/${BRANCH}`);
          state.head = sha;
          return { data: { object: { sha } } };
        }),
      },
      repos: {
        getContent: record("repos.getContent", ({ path, ref }) => {
          assert.equal(path, PATH);
          const sha =
            ref === "master" ? BASE : ref === BRANCH ? state.head : ref;
          if (options.contentError) throw options.contentError;
          if (options.content !== undefined) return { data: options.content };
          assert.ok(state.sourceByCommit[sha], `Unknown content ref: ${ref}`);
          return {
            data: {
              type: "submodule",
              sha: state.sourceByCommit[sha],
              submodule_git_url: "https://github.com/mlflow/skills",
            },
          };
        }),
        compareCommits: record("repos.compareCommits", () => ({
          data: { files: options.changedFiles ?? [{ filename: PATH }] },
        })),
      },
      users: {
        getByUsername: record("users.getByUsername", ({ username }) => {
          assert.equal(username, BOT_NAME);
          return { data: { id: BOT_ID } };
        }),
      },
      pulls: {
        list: async () => {
          throw new Error("Use github.paginate for the PR list");
        },
        create: record("pulls.create", (params) => {
          const pr = { number: 42, html_url: PR_URL, ...params };
          state.prs = [pr];
          return { data: pr };
        }),
        update: record("pulls.update", (params) => {
          assert.equal(params.pull_number, 42);
          if (params.state === "closed") state.prs = [];
          return { data: { number: 42, html_url: PR_URL, ...params } };
        }),
      },
    },
    paginate: async (method, params) => {
      assert.equal(method, github.rest.pulls.list);
      calls.push({ name: "pulls.list", params });
      assert.equal(params.owner, "mlflow");
      assert.equal(params.repo, "mlflow");
      assert.equal(params.state, "open");
      assert.equal(params.head, `mlflow:${BRANCH}`);
      assert.equal(params.base, "master");
      return state.prs;
    },
  };
  const core = {
    info: (message) => logs.push(message),
    warning: (message) => logs.push(message),
  };
  const run = (sourceSha = SOURCE) =>
    sync({ github, core, sourceSha, appSlug: APP_SLUG });
  const getCalls = (name) => calls.filter((call) => call.name === name);
  const onlyCall = (name) => {
    assert.equal(getCalls(name).length, 1, `Expected one ${name} call`);
    return getCalls(name)[0].params;
  };
  const writes = () =>
    calls.filter(({ name }) =>
      /\.(createTree|createCommit|createRef|updateRef|create|update)$/.test(
        name,
      ),
    );
  return { run, calls, getCalls, onlyCall, writes, state, logs };
}

function assertCommit(f, parents) {
  const tree = f.onlyCall("git.createTree");
  assert.equal(tree.base_tree, BASE_TREE);
  assert.deepEqual(tree.tree, [
    { path: PATH, mode: "160000", type: "commit", sha: SOURCE },
  ]);
  const commit = f.onlyCall("git.createCommit");
  assert.equal(commit.tree, CREATED_TREE);
  assert.deepEqual(commit.parents, parents);
  assert.deepEqual(commit.author, { name: BOT_NAME, email: BOT_EMAIL });
  assert.deepEqual(commit.committer, commit.author);
  assert.match(commit.message, /^Update bundled MLflow skills\n\n/);
  assert.ok(commit.message.includes(SOURCE));
  assert.ok(
    commit.message.endsWith(`Signed-off-by: ${BOT_NAME} <${BOT_EMAIL}>`),
  );
}

function assertPrBody(body) {
  assert.equal(typeof body, "string");
  assert.ok(body.includes(SOURCE));
  assert.ok(body.includes("https://github.com/mlflow/skills"));
  assert.match(body, /### What changes are proposed in this pull request\?/);
  assert.match(body, /### How is this PR tested\?/);
  assert.match(body, /### Does this PR require documentation update\?/);
  assert.match(body, /### Does this PR require updating the \[MLflow Skills\]/);
  assert.match(body, /### Release Notes/);
  assert.match(body, /#### Is this PR a critical bugfix or security fix/);
}

test("creates one gitlink-only, signed-off commit and a PR against master", async () => {
  const f = fixture();
  const result = await f.run();
  assert.equal(result.status, "created");
  assert.equal(result.url, PR_URL);
  assertCommit(f, [BASE]);
  assert.equal(f.onlyCall("git.createRef").sha, CREATED_COMMIT);
  assert.equal(f.getCalls("git.updateRef").length, 0);
  const pr = f.onlyCall("pulls.create");
  assert.equal(pr.owner, "mlflow");
  assert.equal(pr.repo, "mlflow");
  assert.equal(pr.base, "master");
  assert.equal(pr.head, BRANCH);
  assertPrBody(pr.body);
});

test("refreshes a pending bot PR with a non-forced fast-forward commit", async () => {
  const f = fixture({ existingHead: true, existingPr: true });
  const result = await f.run();
  assert.equal(result.status, "updated");
  assert.equal(result.url, PR_URL);
  assertCommit(f, [BASE, HEAD]);
  const update = f.onlyCall("git.updateRef");
  assert.equal(update.sha, CREATED_COMMIT);
  assert.equal(update.force, false);
  assert.equal(f.getCalls("git.createRef").length, 0);
  assert.equal(f.getCalls("pulls.create").length, 0);
  assertPrBody(f.onlyCall("pulls.update").body);
  const compare = f.onlyCall("repos.compareCommits");
  assert.equal(compare.base, PREVIOUS_BASE);
  assert.equal(compare.head, HEAD);
});

test("does nothing when master already pins the source and there is no pending PR", async () => {
  const f = fixture({ baseSource: SOURCE });
  assert.equal((await f.run()).status, "up-to-date");
  assert.deepEqual(f.writes(), []);
});

test("rerunning after PR creation does not rewrite the git branch or create a duplicate PR", async () => {
  const f = fixture();
  await f.run();
  const result = await f.run();
  assert.equal(result.status, "up-to-date");
  assert.equal(f.getCalls("git.createCommit").length, 1);
  assert.equal(f.getCalls("git.createRef").length, 1);
  assert.equal(f.getCalls("git.updateRef").length, 0);
  assert.equal(f.getCalls("pulls.create").length, 1);
  assert.equal(f.getCalls("pulls.update").length, 0);
});

test("closes an obsolete bot PR after master reaches the requested source", async () => {
  const f = fixture({
    baseSource: SOURCE,
    existingHead: true,
    existingPr: true,
  });
  assert.equal((await f.run()).status, "closed");
  assert.equal(f.onlyCall("pulls.update").state, "closed");
  assert.equal(f.writes().length, 1);
  assert.equal(f.getCalls("repos.compareCommits").length, 1);
});

test("creates a PR for a valid bot branch left behind by a previous failed run", async () => {
  const f = fixture({
    existingHead: true,
    headSource: SOURCE,
    headParent: BASE,
  });
  assert.equal((await f.run()).status, "created");
  assert.equal(f.getCalls("git.createCommit").length, 0);
  assert.equal(f.getCalls("git.updateRef").length, 0);
  assertPrBody(f.onlyCall("pulls.create").body);
});

test("rebases a pending PR when master advances even if its skills pin is current", async () => {
  const f = fixture({
    existingHead: true,
    existingPr: true,
    headSource: SOURCE,
  });
  assert.equal((await f.run()).status, "updated");
  assertCommit(f, [BASE, HEAD]);
  assert.equal(f.onlyCall("git.updateRef").force, false);
});

test("does not rewrite a current pending PR already based on master", async () => {
  const f = fixture({
    existingHead: true,
    existingPr: true,
    headSource: SOURCE,
    headParent: BASE,
  });
  const result = await f.run();
  assert.equal(result.status, "up-to-date");
  assert.equal(result.sha, HEAD);
  assert.equal(f.getCalls("git.createCommit").length, 0);
  assert.equal(f.getCalls("git.updateRef").length, 0);
  assert.equal(f.getCalls("pulls.create").length, 0);
  assert.deepEqual(f.writes(), []);
});

test("refuses to update an open PR whose bot branch has disappeared", async () => {
  const f = fixture({ existingPr: true });
  await assert.rejects(f.run(), /missing.*branch/);
  assert.deepEqual(f.writes(), []);
});

test("propagates authentication failure while resolving the bot identity", async () => {
  const f = fixture({ errors: { "users.getByUsername": apiError(403) } });
  await assert.rejects(f.run(), { status: 403 });
  assert.deepEqual(f.writes(), []);
});

for (const status of [401, 403, 500]) {
  test(`does not treat a ${status} head lookup failure as a missing branch`, async () => {
    const f = fixture({ headError: apiError(status) });
    await assert.rejects(f.run(), { status });
    assert.deepEqual(f.writes(), []);
  });
}

for (const [name, headCommit] of [
  [
    "human author",
    { author: { name: "Contributor", email: "contributor@example.com" } },
  ],
  [
    "unrecognized commit message",
    { message: `Manual update\n\nSigned-off-by: ${BOT_NAME} <${BOT_EMAIL}>` },
  ],
  [
    "missing sign-off",
    {
      message: `Update bundled MLflow skills\n\nSkills revision: ${OLD_SOURCE}`,
    },
  ],
  [
    "a sign-off without the bot numeric ID",
    {
      message: `Update bundled MLflow skills\n\nSigned-off-by: ${BOT_NAME} <${BOT_NAME}@users.noreply.github.com>`,
    },
  ],
  ["no parent commit", { parents: [] }],
]) {
  test(`refuses to modify a branch with ${name}`, async () => {
    const f = fixture({ existingHead: true, existingPr: true, headCommit });
    await assert.rejects(f.run());
    assert.deepEqual(f.writes(), []);
  });
}

test("refuses an existing branch containing unrelated file changes", async () => {
  const f = fixture({
    existingHead: true,
    existingPr: true,
    changedFiles: [{ filename: PATH }, { filename: "README.md" }],
  });
  await assert.rejects(f.run());
  assert.deepEqual(f.writes(), []);
});

test("refuses an existing branch that does not contain a skills pointer change", async () => {
  const f = fixture({ existingHead: true, existingPr: true, changedFiles: [] });
  await assert.rejects(f.run());
  assert.deepEqual(f.writes(), []);
});

test("does not close a human-modified PR when master is already current", async () => {
  const f = fixture({
    baseSource: SOURCE,
    existingHead: true,
    existingPr: true,
    headCommit: {
      author: { name: "Contributor", email: "contributor@example.com" },
    },
  });
  await assert.rejects(f.run());
  assert.deepEqual(f.writes(), []);
});

test("propagates a branch update race without forcing or refreshing the PR", async () => {
  const f = fixture({
    existingHead: true,
    existingPr: true,
    errors: { "git.updateRef": apiError(422, "Update is not a fast forward") },
  });
  await assert.rejects(f.run(), { status: 422 });
  assert.equal(f.onlyCall("git.updateRef").force, false);
  assert.equal(f.getCalls("pulls.update").length, 0);
  assert.equal(f.getCalls("pulls.create").length, 0);
});

for (const sourceSha of [
  "",
  "b".repeat(39),
  "b".repeat(41),
  "B".repeat(40),
  "../master",
  undefined,
]) {
  test(`rejects invalid source SHA ${JSON.stringify(sourceSha)} before API calls`, async () => {
    const f = fixture();
    await assert.rejects(
      sync({ github: {}, core: {}, sourceSha, appSlug: APP_SLUG }),
      /commit SHA/,
    );
    assert.deepEqual(f.calls, []);
  });
}

test("propagates a missing target submodule without creating a branch", async () => {
  const f = fixture({ contentError: apiError(404) });
  await assert.rejects(f.run(), { status: 404 });
  assert.deepEqual(f.writes(), []);
});

for (const [name, content] of [
  ["a directory", []],
  ["a regular file", { type: "file", sha: OLD_SOURCE }],
  [
    "the wrong repository",
    { sha: OLD_SOURCE, submodule_git_url: "https://github.com/other/skills" },
  ],
]) {
  test(`refuses to change the target path when it is ${name}`, async () => {
    const f = fixture({ content });
    await assert.rejects(f.run());
    assert.deepEqual(f.writes(), []);
  });
}

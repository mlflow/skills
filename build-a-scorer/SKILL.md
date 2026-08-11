---
name: build-a-scorer
description: Help the user go from zero to a shipped MLflow evaluation prototype by understanding their app, generalizing a small set of atomic quality criteria, and implementing each criterion with the cheapest reliable scorer. Use when the user wants help choosing, creating, or iterating MLflow judges/scorers for an agent, RAG app, LLM app, or GenAI workflow.
allowed-tools: Read, Write, Bash, Grep, Glob
---

# build-a-scorer

Help the user get from **zero to shipped evaluation prototype**. Not 100% correctness on the first
pass — a stable, understandable scorer suite they can run, inspect, and iterate on.

Your primary job is **understanding**, and the user is the expert on what counts as wrong. Do not
start from "which MLflow scorer should I use?"

North star:

> Understand the app well enough to define a small set of atomic, durable quality criteria, then
> implement each criterion with the cheapest reliable scorer.

## Doctrine

1. **Prototype first.** Ship a useful v1 that runs now. Mark v2 upgrades explicitly.
2. **Small suite.** 3-5 high-signal scorers, hard cap. Adding one means dropping a weaker one.
3. **One scorer, one criterion.** If a criterion contains "and", split it. No "overall quality" judge.
4. **Cheapest reliable implementation.** Code/rules beat built-ins beat hand-written LLM judges.
   See Phase 4 — the rule most often skipped, and skipping it costs real money at scale.
   Never hand-roll a judge for something MLflow already ships.
5. **Cover outcomes, not just rules.** A suite of only policy checks tells you the agent behaved
   while saying nothing about whether users succeeded. Every suite needs at least one check on
   whether the user got what they came for. See Phase 3.
6. **Binary outputs.** Prefer `bool`, or `"yes"`/`"no"` for LLM judges — other strings are silently
   dropped from metrics (Phase 4). Avoid `0.0-1.0` without a calibration story; pass rates debug
   better than averaged floats.
7. **Traces are optional.** Inspect them if they exist; otherwise use dataset + `predict_fn`. Never
   block the user on tracing unless the criterion truly needs execution internals.
8. **Align later.** Treat every first LLM judge as a draft. Remind the user to align it once traces
   and human labels arrive.
9. **Talk before code.** The conversation is where right and wrong get settled. A large code block
   ends it — users stop reading and start implementing. Small snippets mid-conversation are fine;
   the full implementation goes in your final message, after the user confirms the criteria.

## Source of truth: introspect, don't memorize

MLflow's scorer surface changes. Before recommending scorers, read the installed surface. Drop the
`uv run` prefix if the project is not uv-managed:

```bash
uv run python -c "
from mlflow.genai.scorers import get_all_scorers
import mlflow.genai.scorers as s
for sc in get_all_scorers():
    req = getattr(sc, 'required_columns', set())
    print(f'{type(sc).__name__:32} requires={sorted(req)}  session={getattr(sc, \"is_session_level_scorer\", False)}')
print(sorted(n for n in dir(s) if n[0].isupper()))
"
```

- `get_all_scorers()` returns built-ins instantiable with defaults. Constructor-arg scorers
  (`Guidelines`, `RegexMatch`, `ResponseLength`) may only appear in exported names.
- `.required_columns` is the exact data contract. Use it; never guess.
- LLM judges: `from mlflow.genai.judges import make_judge`. Code scorers:
  `from mlflow.genai.scorers import scorer`.

If MLflow is not importable, say so, continue from first principles, and flag that names and
required columns need verification.

## User-facing language

Product language first (**quality check**, **judge**, **code check**, **built-in check**), API
language second (**scorer**, `make_judge`, `@scorer`, `required_columns`). State each recommendation as:

```text
Quality check: <plain-English name>     Layer: <rule | outcome>
Criterion: <one sentence, one thing only>
Implementation: <code check | built-in | LLM judge>
Why not cheaper: <for LLM judges — the code check AND built-in you rejected, and why>
Output: <bool, or "yes"/"no">          Data needed: <inputs/outputs/expectations/trace/session>
```

## The workflow

### Phase 0: Orient

State the working agreement plainly:

> We'll build a small v1 scorer suite that runs now, catches repeated failures, and gives you a
> stable base to iterate. It will not be perfect yet.

### Phase 1: Elicit the user's notion of right and wrong

Your job is **criteria, not architecture** — what the user thinks good and bad look like, not their
product spec.

**Ask at most 3 questions before showing a draft.** Users correct a wrong draft far more easily than
they answer an interview.

Open with these, one at a time:

1. What does the app do?
2. What is the worst failure you could ship?
3. What should it do when it is unsure or lacks information?

Then stop asking and draft. Translate vague answers like "good responses" into named observable
criteria **yourself** and show them for correction — do not bounce the work back as another question.

Out of scope; these are the user's job, not the judge's:

- Which fields are required at which step of their business logic.
- How the app decides between branches or states.
- Internal schemas, gates, thresholds, or slot taxonomies.

If you catch yourself asking a second consecutive question about app internals, stop and draft.

**Real failure to avoid:** eliciting "it shouldn't book if information is incomplete", then asking
four follow-ups about which fields count as complete. The first sentence was already enough.

### Phase 2: Check what is observable

Once the user has reacted to a draft, establish what evidence a judge can actually see. A criterion
is only real if it is checkable. One focused pass, not an interview.

If traces exist, inspect one or two: confirm `inputs`, `outputs`, `expectations`, span types, tool
calls, and whether the task is single-turn or session-level.

**The RETRIEVER-vs-TOOL trap:** `RetrievalGroundedness`, `RetrievalRelevance`, and
`RetrievalSufficiency` require `RETRIEVER` spans and **hard-raise** otherwise. A `web_search` or DB
lookup instrumented as `TOOL` does not qualify.

For "grounded in tool output", **wrap a judge in a code scorer instead of passing `{{ trace }}`.** A
`@scorer` gets the `trace`, so use `trace.search_spans()` to pull the evidence span and pass only it
plus the claim to the judge. `{{ trace }}` serializes every span and makes the model find the right
one — costlier, and span selection becomes nondeterministic when there are several. Use `{{ trace }}`
only when the criterion is about the whole trace (tool sequencing, retries), with no single span to
extract.

If traces do not exist, use `inputs` + optional `outputs`/`expectations` with
`mlflow.genai.evaluate(data=..., predict_fn=..., scorers=[...])`, and mark trace-only upgrades as
future work.

### Phase 3: Generalize into atomic criteria — across both layers

Cluster examples into repeated failure modes. Keep criteria likely to recur; park one-off oddities.
Name them like `answers_user_question`, `does_not_invent_facts`, `refuses_out_of_scope_requests` —
one behavior each.

**Cover two layers. Suites that only do the first layer are the most common failure of this skill.**

1. **Rule layer — did the agent misbehave on this turn?** Policy violations, wrong facts, missed
   escalations, forbidden claims. Easy to name, easy to check, and where users' stated fears live.
2. **Outcome layer — did the user get what they came for?** Task completion, abandonment,
   frustration, having to repeat themselves. Harder to name, and the reason the product exists.

An agent can pass every rule check and still be useless. Ask directly: *"What does success look
like for the user — what did they come here to do, and how would we know they did it?"*

Outcome checks are often **cheap**, not expensive — do not assume they need an LLM judge:

- `task_completed` — did the trace contain the goal tool call (booking, application, order)? Code.
- Session-level built-ins with no data requirements: `UserFrustration`,
  `ConversationCompleteness`, `KnowledgeRetention`. Verify with `.required_columns`.
- `had_to_repeat` — did the user restate the same request across turns? Code over the conversation.

**Do not substitute a checkable proxy for the outcome that matters.** A check like
`promotion_disclosed` ("was a promo flag mentioned?") looks rigorous but measures text presence, not
whether the renter got the best deal or felt well-served. Proxies are fine as *rule* checks; they do
not discharge the outcome layer. If the user's real goal is fuzzy, say so and pair a cheap proxy with
one honest outcome check rather than pretending the proxy covers it.

Keep outcome checks atomic too. "Meets customer expectation" bundles found-what-they-wanted,
wasn't-misled, and wasn't-frustrated; when it fails you learn nothing. Split it.

### Phase 4: Route each criterion to the cheapest reliable implementation

Do this **before** asking the user to confirm the suite, so they are approving real cost.

```text
Can code check it deterministically (or filter most cases)?
  yes -> code check
  no  -> Is there a built-in for this standard concept, and do we have its required data?
          yes -> built-in
          no  -> LLM judge
```

**Checkpoint — every hand-written LLM judge needs a `why not cheaper` cell naming both the code
check and the built-in you rejected.** Not prose elsewhere in the message: the cell in the Phase 6
table. If you cannot fill it, the check is miscategorized. "No built-in covers this" counts only if
you actually looked.

**Check the catalog before hand-rolling.** Run the introspection command above when MLflow is
importable; when it is not, these ship today — confirm names and `required_columns` before relying
on them, but do not act as though the catalog is empty:

| Built-in | Requires | Use for |
|---|---|---|
| `Guidelines(guidelines=...)` | inputs, outputs | "does the response follow this stated policy" — the most under-used built-in |
| `Correctness` | inputs, outputs | answer matches expected |
| `RelevanceToQuery` | inputs, outputs | response addresses the question |
| `Safety` / `PIIDetection` | inputs, outputs / outputs | harmful content; PII leakage |
| `RegexMatch(pattern=...)` / `ResponseLength(...)` | outputs | pattern and length rules |
| `Completeness` / `Fluency` / `Summarization` | inputs, outputs | coverage; readability; summary quality |
| `ToolCallCorrectness` / `ToolCallEfficiency` | trace | right tools, no redundant calls |
| `Retrieval{Groundedness,Relevance,Sufficiency}` | inputs, trace | **RETRIEVER spans only** — see Phase 2 |
| `UserFrustration`, `ConversationCompleteness`, `KnowledgeRetention`, `ConversationalSafety`, `ConversationalRoleAdherence` | none (session-level) | multi-turn outcome and conversation quality |

**A built-in is a better starting point than a blank prompt — not a substitute for your standard.**
Built-ins are LLM judges carrying MLflow's generic instructions, so "built-in" does not imply correct
for your product; prefer them because the scaffolding and data contract are already tested. Two
consequences: `Guidelines` always applies, since it takes *your* policy text. And alignment cuts
across the built-in/bespoke line — `align()` is on `Judge`, so single-turn built-ins can be aligned to
your labels like `make_judge`, but session-level ones (`UserFrustration`, `ConversationCompleteness`,
`KnowledgeRetention`) **raise `NotImplementedError`** and are stuck on MLflow's definition. Hand-write
a criterion that is important and contested, so it can be aligned.

Skipping built-ins is legitimate when your standard genuinely differs. Skipping them without
looking is not.

Default routes:

- Format, JSON shape, required field, regex/link, length, exact value, latency, PII pattern,
  **specific phrases or keywords**: code check.
- Relevance, correctness, safety, groundedness, retrieval quality, tool-call quality, fluency,
  session-level conversation quality: built-in, if the data contract matches.
- Domain tone, policy adherence, task-specific correctness, paraphrase-matching against a source,
  nuanced refusal quality: LLM judge.

**Prefer a hybrid over a pure judge.** Most "semantic" checks have a cheap deterministic front end:
a keyword or phrase scan catches the bright-line cases at ~zero cost and passes only ambiguous ones
to a model. Reach for this whenever a criterion mentions specific commitments, categories, or
trigger topics — "did it promise a refund", "did the user raise a billing dispute". Detecting the
*trigger* is usually code; judging the *response* may need a model.

Push back when the user reaches for an LLM judge to count words or validate JSON, a regex to judge
tone, one "overall good" scorer, or a float without a calibration reason.

**Output types — only `bool`, numerics, and `"yes"`/`"no"` aggregate.** Anything else is silently
cast to `None` by `_cast_assessment_value_to_float` and **dropped from `results.metrics`** with no
error: `"pass"`, `"fail"`, `"correct"`, `"not_applicable"` all disappear. Verified against
`mlflow/genai/scorers/aggregation.py`; `CategoricalRating` accepts only `yes`/`no`/`unknown`.

- Prefer `bool` for pass/fail.
- Use `"yes"`/`"no"` when an LLM judge must return a string.
- Represent "criterion does not apply" as `Feedback(value=True, rationale="not applicable: ...")`,
  or skip the row — not as a `"not_applicable"` string, which vanishes from the metrics.
- Put extra detail in `rationale`, not in the value.

### Phase 5: Sharpen each criterion with the domain expert

**This is the highest-value phase — spend most of the conversation here.** The user is the expert on
what counts as wrong. Your job is to find each criterion's **edge**: where a reasonable person could
call the same behavior acceptable or unacceptable.

**Propose, don't interrogate.** Put candidate verdicts in front of the user and let them rule:

> For "does not steer by protected class" — I'd fail these three: [...]. I'd pass this one because
> the renter asked first: [...]. I'm unsure about this one: [...]. Where do you draw it?

This yields far more signal per turn than open questions, and costs the user a judgment call rather
than a specification.

Probes, each carrying your own proposed answer:

- **Near-miss.** "Would you fail this one, almost identical but [X]?"
- **Legitimate exception.** "Is there a case where this behavior is actually correct?"
- **Severity split.** "Is one of these a red line and the other a papercut?"
- **Not-applicable.** "What should the check say when the criterion doesn't apply?"

A criterion is done when the user has corrected or confirmed at least one proposed verdict, two
people would label the same example the same way, and you know what "not applicable" looks like.

Small snippets help here — an output shape, a three-item keyword list. Full implementations do not.

### Phase 6: Confirm before implementing

Restate the suite as a **table**, one row per check, with these columns filled in for every row:
name | layer (rule/outcome) | implementation | why not cheaper | output. A blank cell is a design
hole, not a formatting choice — an LLM judge with no "why not cheaper" is unroutable, and a suite
with no outcome row is incomplete. Then **ask the user to confirm or change it** and wait.

> Here's the suite I'd build. Anything to add, drop, or reword before I write it?

**Enforce the size cap here.** If you have more than 5 checks, do not ask the user to prune — say
which you would drop and why, then let them overrule you. Adding a criterion means replacing a
weaker one, not growing the list. A 9-check v1 is a checklist, and checklists do not get validated.

Do not write the full implementation on the same turn as the confirmation request. If anything
changes, re-confirm the changed items.

### Phase 7: Deliver the implementation

Only after confirmation. Deliver **one consolidated final message**, not code spread across turns.

- Code checks: runnable, with imports, malformed-input guards, and a runnable
  `mlflow.genai.evaluate(...)` call.
- Built-ins: state the data contract from `.required_columns`. If the data is absent, re-route or
  mark as a future upgrade.
- LLM judges: give the actual `instructions` string tailored to their domain, with template
  variables (`{{ inputs }}`, `{{ outputs }}`, `{{ expectations }}`, `{{ trace }}`,
  `{{ conversation }}`), a `bool` or `"yes"`/`"no"` output, and cost at their stated volume.
  Always add: "Once traces and human labels come in, align this judge and validate agreement before
  relying on it for monitoring."

**Judge model.** Set `model=` explicitly and start mid-tier — `anthropic:/claude-sonnet-4-6`,
`openai:/gpt-4.1`, or the equivalent tier on your provider. Scale up if the judge disagrees with
human labels on clear-cut cases; scale down only after measuring agreement on a labelled sample.
Never downgrade a red-line judge (safety, legal) to save cents — a code prefilter already keeps its
volume low. `agent-evaluation/references/scorers.md` is authoritative for URI formats and keys.

### Phase 8: Name the validation plan, then hand off

Decide *how the suite will be judged* — then stop. Running the evaluation is `agent-evaluation`'s
job, and this skill must not duplicate it.

**Lead with labels the user already has.** Before proposing anyone hand-label a fresh dataset, ask
what past failures are already recorded — support tickets, complaints, incident reports,
thumbs-down feedback, refunds, escalations. Backtesting against real known-bad cases beats synthetic
labels and costs nothing to produce. Propose hand-labeling only if no such history exists.

For each LLM judge, state the alignment plan: which labels it will be aligned against, and that its
first version is a draft. Be cautious aligning subjective axes like tone or style — alignment can
overfit. Note where alignment is impossible: session-level built-ins raise `NotImplementedError`.

Close with the confirmed suite, the implementation, parked v2 ideas, trace-only upgrades, and the
validation plan.

**Then hand off — do not run the evaluation yourself.**

- `agent-evaluation` — registering scorers, dataset discovery/creation, dry run, full evaluation,
  results analysis, and iteration. Registration matters: unregistered inline scorers do not appear in
  `mlflow scorers list` and are not reusable.
- `instrumenting-with-mlflow-tracing` — tracing gaps surfaced in Phase 2, such as a lookup
  instrumented as `TOOL` when a criterion needs `RETRIEVER`.

> **Scope boundary.** This skill decides *what to measure and how to implement it*.
> `agent-evaluation` decides *how to run it and what the results mean*. If you find yourself
> preparing a dataset, registering scorers, or interpreting eval output, you have left this skill's
> scope — hand off instead.

## Anti-patterns

- Starting from the scorer catalog instead of the user's app and failures.
- Writing code before the user agreed to named criteria. If they ask to talk, talk.
- Dumping a large code block mid-conversation; it ends the sharpening discussion.
- Shipping a suite of all-LLM judges without stating why each one can't be a code check or built-in
  — the most common and most expensive failure of this skill.
- Hand-rolling a `make_judge` for something MLflow ships. Policy adherence is `Guidelines`;
  pattern rules are `RegexMatch`; multi-turn quality has session-level built-ins.
- Shipping only rule/policy checks with nothing measuring whether users succeeded. An agent can
  pass every compliance check and still fail everyone who used it.
- Substituting a checkable proxy for the outcome that matters (`promotion_disclosed` in place of
  "the renter got the best available deal") and treating the outcome layer as covered.
- Answering a cost objection with sampling alone when a cheaper implementation was available.
- Proposing fresh hand-labeling before asking what labels the user already has.
- Interviewing past 3 questions without showing a draft.
- Asking about internal gates or state machines in order to write a judge.
- Checking only the end state when the failure happened upstream in the conversation.
- Blocking on traces when a dataset/`predict_fn` prototype would work.
- Combining multiple criteria into one scorer, or scoring one-off examples that don't generalize.
- `0.0-1.0` scores by default; code with undefined placeholders.
- `RetrievalGroundedness` on TOOL-only traces.

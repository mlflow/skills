![MLflow Skills](/assets/header.svg)

# MLflow Skills for Coding Agents

> Turn your favorite coding agent into an LLMOps expert with MLflow skills.

Build, debug, and evaluate GenAI applications with confidence. These skills give your AI coding assistant deep knowledge of MLflow's tracing, evaluation, and observability features.

Works with any coding agent that support Skills, including [Claude Code](https://docs.anthropic.com/en/docs/claude-code), [Cursor](https://www.cursor.com/), [Codex CLI](https://github.com/openai/codex), [Gemini CLI](https://github.com/google-gemini/gemini-cli), and [OpenCode](https://github.com/opencode-ai/opencode).

## Why MLflow Skills?

Building production-ready AI agents is hard. You need observability to understand what your agent is doing, evaluation to measure quality, and debugging tools when things go wrong. MLflow provides SDKs and best practices for all of these operations, and with skills we bring them directly into the environment where LLM agent development happens. Now you can go to your favorite coding agent and just ask:

- *"Add tracing to my LangChain app"* → Instruments your code automatically
- *"Why did this trace fail?"* → Analyzes spans, finds root causes, suggests fixes
- *"Evaluate my agent's accuracy"* → Sets up datasets, scorers, and runs evaluation
- *"Improve my agent and verify your work"* → Gives the coding agent reproducible and verifiable mechanism with MLflow eval to hill climb on quality
- *"Show me token usage trends"* → Queries metrics and analyze trends

---

## Available Skills

These are three workflow groups, not new skills. Existing skill names and install paths stay the same.

| Owner | Responsibilities | Existing skills |
|-------|------------------|-----------------|
| **Setup** | Select and verify the backend only when configuration is missing; reuse working environments. | `mlflow-onboarding`, `sagemaker-mlflow` |
| **Tracing** | Instrument, retrieve and inspect traces/sessions; diagnose failures; query operational metrics. | `instrumenting-with-mlflow-tracing`, `retrieving-mlflow-traces`, `analyze-mlflow-trace`, `analyze-mlflow-chat-session`, `debug-agent`, `querying-mlflow-metrics` |
| **Evaluation** | Define quality criteria, build/align scorers, prepare datasets, evaluate and verify behavior changes. | `agent-evaluation`, `build-a-scorer`, `fix-agent-issue` |

`mlflow-agent` routes requests across these groups. `searching-mlflow-docs` supports all three.
Scorer design and behavior fixes are evaluation workflows; diagnosis without a code change belongs to tracing.

Start with the requested task. If Databricks tooling already provides a working MLflow environment,
reuse its tracking URI, credentials and experiment. Setup is a conditional prerequisite, not a step
to repeat before every tracing or evaluation task.

---

## Installation

### Using `skills` installer

```bash
npx skills add mlflow/skills
```

### Direct Installation from Source

```bash
git clone https://github.com/mlflow/skills.git
cp -r mlflow-skills/* ~/.claude/skills/
```

Change the `~/.claude/skills/` directory to the appropriate location for your coding agent, e.g., `~/.codex/skills/` for Codex.

### Project-Level Installation

Add skills to your project for team sharing:

```bash
cd your-project
git clone https://github.com/mlflow/skills.git .skills/mlflow
# Or as a submodule:
git submodule add https://github.com/mlflow/skills.git .skills/mlflow
```

---

## Auto-Suggestion Hook (Optional)

The `hooks/` directory contains a `UserPromptSubmit` hook that automatically detects MLflow-related patterns in your prompts and surfaces the right skill before the agent responds — no need to remember which skill does what.

### Install the Hook

**Step 1:** Copy the hook somewhere permanent:

```bash
cp hooks/mlflow-suggest-hook.py ~/.claude/hooks/mlflow-suggest-hook.py
```

**Step 2:** Add it to your Claude Code settings (`~/.claude/settings.json`):

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "type": "command",
        "command": "python3 ~/.claude/hooks/mlflow-suggest-hook.py"
      }
    ]
  }
}
```

**Step 3:** Start a new session. When you ask something like *"Add tracing to my app"*, you'll see:

```
💡 Use the `instrumenting-with-mlflow-tracing` skill to add MLflow tracing.
```

See [`hooks/README.md`](hooks/README.md) for the full keyword-to-skill mapping and troubleshooting.

---

## Quick Examples

### Instrument Your App with Tracing

```
> Add MLflow tracing to my OpenAI app

The coding agent will:
1. Detect your LLM framework
2. Add the right autolog call
3. Configure experiment tracking
4. Verify traces are being captured
```

### Debug a Failed Trace

```
> Analyze trace tr-abc123 — why did it return the wrong answer?

The coding agent will:
1. Fetch the trace and examine all spans
2. Check assessments for quality signals
3. Walk the span tree to find the failure point
4. Correlate with your codebase to identify root cause
5. Suggest specific fixes
```

### Evaluate Agent Quality

```
> Set up evaluation for my customer support agent

The coding agent will:
1. Discover existing evaluation datasets
2. Help create test cases if needed
3. Select appropriate scorers (correctness, relevance, etc.)
4. Run evaluation and analyze results
5. Generate improvement recommendations
```

### Query Usage Metrics

```
> Show me token usage trends for the last 7 days

The coding agent will:
1. Query the MLflow metrics API
2. Generate time-series breakdowns
3. Identify cost spikes or anomalies
4. Suggest optimization opportunities
```

---

## Requirements

- **MLflow 3.8+** (Instructions and code examples are tested with MLflow 3.8)

---

## Contributing

We welcome contributions! Here's how to help:

1. **Report issues** — Found a skill giving incorrect advice? [Open an issue](https://github.com/mlflow/mlflow-skills/issues)
2. **Suggest improvements** — Have ideas for better workflows? We'd love to hear them
3. **Add new skills** — See [skill authoring best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices) and file a pull request

### Testing skill behavior

The YAML scenarios in `tests/configs/` run a real Claude Code participant in a fresh project and Claude home, install the selected repository skills plus the MLflow tracing plugin, and verify its work against a real MLflow server. Each scenario checks both a successful skill read and an observable task outcome. Merely mentioning an API or reporting success does not pass.

Requires an authenticated `claude` CLI, Git, Node/npm (for Claude tracing and TypeScript), and `uv`. Run a scenario:

```sh
uv run --with mlflow --with litellm --with pytest --with langgraph --with langchain-core \
  python tests/test_skill.py tests/configs/agent_evaluation.yaml
```

| Skill/workflow | Scenarios | Observable checks |
|---|---|---|
| Onboarding | `onboarding_existing`, `onboarding_unavailable`, `onboarding_classic_ml` | Reuse configuration and experiment; persist smoke trace; report a real connection failure without resetting setup; reload an actual sklearn model and verify held-out predictions |
| Dispatcher | `dispatcher_scorer` | Load the dispatcher and scorer skill; execute a scorer against held-out empty/missing/null/valid answers |
| Documentation | `docs_classic_ml` | Fetch official Markdown docs; cite a real executable classic-ML example |
| SageMaker | `sagemaker_prerequisites` | Execute prerequisite checks; stop when plugin/credentials are missing |
| Instrumentation | `tracing_test`, `tracing_typescript`, `langgraph_autolog_guidance` | Persist exact Python/TypeScript inputs and outputs; real graph/tool/model spans without redundant tracing |
| Trace retrieval | `trace_retrieval` | Exact matching IDs amid time, status and tag distractors |
| Single-trace diagnosis | `trace_analysis` | Identify the actual failing child span and its timeout inputs |
| Session diagnosis | `session_analysis` | Ordered turns, first incorrect answer, propagation, evidence, and failed-scorer exclusion |
| Aggregate metrics | `trace_metrics` | Exact counts and token sums from the real metrics endpoint; distinguish unsupported endpoints from other failures |
| Diagnosis versus repair | `diagnosis_only`, `agent_debug_trace_first` | Diagnosis preserves source; repair follows trace evidence and verifies refund/general routing |
| Evaluation | `agent_evaluation`, `evaluation_precomputed`, `evaluation_dry_run` | Nested predictor kwargs; exact six-row scores and mean; reuse dataset/scorer; never rerun precomputed outputs; stop after failed dry run |
| Scorer design | `scorer_deterministic_suite`, `scorer_model_criteria` | Execute independent criteria on private malformed/near-miss cases; code, built-in and domain judges; preserve model selection |
| Judge alignment | `scorer_label_alignment`, `scorer_memalign` | Criterion-specific human labels and train/holdout splits; missing-provider blocker; real MemAlign plus private held-out agreement |

Model-backed scenarios require working provider credentials. Set `TEST_JUDGE_MODEL` for `scorer_model_criteria`; for `scorer_memalign`, also set `TEST_EMBEDDING_MODEL` and add `--with dspy`. The repair scenario's existing LLM judge accepts `JUDGE_MODEL`. No credentials or private endpoint names belong in scenario files. Each run retains `environment.json`, `skill-inventory.json`, `verification.json`, participant transcripts, generated files, and the local MLflow store in its printed work directory. Setup failures and skill failures both return nonzero, with different exit codes.

To run the local scenarios sequentially, excluding those needing a model provider:

```sh
for config in tests/configs/*.yaml; do
  case "$config" in
    *scorer_model_criteria*|*scorer_memalign*|*agent_debug_trace_first*) continue ;;
  esac
  uv run --with mlflow --with litellm --with pytest --with langgraph --with langchain-core \
    python tests/test_skill.py "$config" || exit 1
done
```

Supporting verifier regression checks run without Claude or model calls:

```sh
uv run --with mlflow --with pytest pytest -q tests/test_artifact_judges.py
```

These scenarios are behavioral regression coverage, not a claim of exhaustive platform coverage or a baseline-versus-treatment comparison. Live SageMaker connection, Databricks UC storage/Review App/online monitoring, GEPA prompt optimization, distributed tracing, and every supported framework still require dedicated environments and scenarios. A successful local run does not verify those integrations. The repository currently has no CI workflow that launches the paid Claude scenarios automatically.

### Creating a New Skill

We recommend using the [skill-creator](https://github.com/anthropics/skills/tree/main/skills/skill-creator) from Anthropic's skills repository as a starting point.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

## Links

- 📖 [Read MLflow Documentation](https://mlflow.org/docs/latest/)
- 📖 [Read MLflow Skills Blog](https://mlflow.org/blog/evaluating-skills-mlflow))
- ⭐️ [Star MLflow on GitHub](https://github.com/mlflow/mlflow)

---

<p align="center">
  <sub>Built with care for the MLflow community</sub>
</p>

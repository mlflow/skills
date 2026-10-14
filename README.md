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

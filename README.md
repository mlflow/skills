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

### Observability & Debugging

| Skill | Description |
|-------|-------------|
| **instrumenting-with-mlflow-tracing** | Instruments Python and TypeScript code with MLflow Tracing. Supports OpenAI, Anthropic, LangChain, LangGraph, LiteLLM, and more. |
| **analyze-mlflow-trace** | Debugs issues by examining spans, assessments, and correlating with your codebase. |
| **analyze-mlflow-chat-session** | Debugs multi-turn chat conversations by reconstructing session history and finding where things went wrong. |
| **retrieving-mlflow-traces** | Powerful trace search and filtering by status, session, user, time range, or custom metadata. |

### Evaluation & Metrics

| Skill | Description |
|-------|-------------|
| **agent-evaluation** | End-to-end agent evaluation workflow — dataset creation, scorer selection, evaluation execution, and results analysis. |
| **querying-mlflow-metrics** | Fetches aggregated metrics (token usage, latency, error rates) with time-series analysis and dimensional breakdowns. |

### Helping New Users

| Skill | Description |
|-------|-------------|
| **mlflow-onboarding** | Guides new users through MLflow setup based on their use case (GenAI apps vs traditional ML). |
| **searching-mlflow-docs** | Searches official MLflow documentation efficiently using the llms.txt index. |

---

## Installation

### Using `skills` installer

```bash
npx skills add mlflow/skills
```

### Direct Installation from Source

```bash
git clone https://github.com/mlflow/mlflow-skills.git
cp -r mlflow-skills/* ~/.claude/skills/
```

Change the `~/.claude/skills/` directory to the appropriate location for your coding agent, e.g., `~/.codex/skills/` for Codex.

### Project-Level Installation

Add skills to your project for team sharing:

```bash
cd your-project
git clone https://github.com/mlflow/mlflow-skills.git .skills/mlflow
# Or as a submodule:
git submodule add https://github.com/mlflow/mlflow-skills.git .skills/mlflow
```

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

- **MLflow 3.9+** (Instructions and code examples are tested with MLflow 3.9)

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
- ⭐️ [Star MLflow on GitHub](https://github.com/mlflow/mlflow)

---

<p align="center">
  <sub>Built with care for the MLflow community</sub>
</p>

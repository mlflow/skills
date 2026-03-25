# MLflow Skills Hooks

Hooks extend your coding agent with automatic skill suggestions based on what you're asking about.

## `mlflow-suggest-hook.py`

A `UserPromptSubmit` hook that detects MLflow-related patterns in your prompts and suggests the most relevant skill before the agent responds. This removes the need to know which skill to invoke — the agent surfaces the right one automatically.

### What It Detects

| Keywords in your prompt | Suggested skill |
|-------------------------|----------------|
| trace, tracing, autolog, span, instrument | `instrumenting-with-mlflow-tracing` |
| evaluat, scorer, judge, dataset, assess, improve quality | `agent-evaluation` |
| trace id, debug trace, why did, what went wrong, analyze trace | `analyze-mlflow-trace` |
| session, conversation, chat history, multi-turn | `analyze-mlflow-chat-session` |
| search traces, find traces, filter traces, get trace | `retrieving-mlflow-traces` |
| metrics, token usage, latency, cost, usage trend | `querying-mlflow-metrics` |
| get started, set up mlflow, onboard, quickstart | `mlflow-onboarding` |
| mlflow docs, mlflow api, how to use mlflow | `searching-mlflow-docs` |

Multiple suggestions can fire in a single prompt if multiple patterns match.

### Installation

**Step 1: Copy the hook to a permanent location**

```bash
# Option A: copy into your home skills directory
cp hooks/mlflow-suggest-hook.py ~/.claude/hooks/mlflow-suggest-hook.py

# Option B: use an absolute path to the hook in-place (see Step 2)
```

**Step 2: Register the hook in your Claude Code settings**

Add the following to your `~/.claude/settings.json` (global) or `.claude/settings.json` (project-level):

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "type": "command",
        "command": "python3 /path/to/hooks/mlflow-suggest-hook.py"
      }
    ]
  }
}
```

Replace `/path/to/hooks/mlflow-suggest-hook.py` with the absolute path where you placed the file.

**Step 3: Verify**

Start a new Claude Code session and type a prompt containing `tracing` or `evaluate`. You should see a suggestion like:

```
💡 Use the `instrumenting-with-mlflow-tracing` skill to add MLflow tracing.
```

### How It Works

Claude Code calls `UserPromptSubmit` hooks before passing your prompt to the model. This hook reads the prompt from stdin as JSON (`{"prompt": "..."}`), performs keyword matching, and prints suggestions to stdout. Claude Code displays these suggestions to you before the agent responds.

The hook exits cleanly with no output when no MLflow patterns are detected, so it has zero impact on non-MLflow conversations.

### Compatibility

This hook works with any coding agent that supports the `UserPromptSubmit` hook protocol, including Claude Code. Other agents may use different hook registration mechanisms — check your agent's documentation for the equivalent setting.

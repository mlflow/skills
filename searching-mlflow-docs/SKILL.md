---
name: searching-mlflow-docs
description: Searches and retrieves MLflow documentation from the official docs site. Use when the user asks about MLflow features, APIs, integrations (LangGraph, LangChain, OpenAI, etc.), tracing, tracking, or requests to look up MLflow documentation. Triggers on "how do I use MLflow with X", "find MLflow docs for Y", "MLflow API for Z".
---

# MLflow Documentation Search

## Workflow

1. Fetch `https://mlflow.org/docs/latest/llms.txt` to find relevant page paths
2. Fetch the `.md` file at the identified path
3. Present results with verbatim code examples

## Step 1: Fetch llms.txt Index

```
WebFetch(
  url: "https://mlflow.org/docs/latest/llms.txt",
  prompt: "Find links or references to [TOPIC]. List all relevant URLs."
)
```

## Step 2: Fetch Target Documentation

Use the path from Step 1, always with `.md` extension:

```
WebFetch(
  url: "https://mlflow.org/docs/latest/[path].md",
  prompt: "Return all code blocks verbatim. Do not summarize."
)
```

## Anti-Patterns

**Do not use `.html` files** — Fetch `.md` source files only.

**Do not use WebSearch** — Always start from llms.txt; web search returns outdated or third-party content.

**Do not use vague prompts** — "Extract complete documentation" allows summarization. Use "Return all code blocks verbatim. Do not summarize."

**Do not use versioned paths** — Always use `/docs/latest/`, never `/docs/3.8/` or other versions unless the user explicitly requests a specific version.

**Do not guess URLs** — Always verify paths exist in llms.txt before fetching. Never construct documentation paths from assumptions.

**Do not follow external links or mix sources** — Stay within `mlflow.org/docs`; do not combine examples with GitHub, PyPI, or third-party documentation.

**Classic ML topics** — Search the index first; it includes classic ML as well as GenAI. If the topic is absent, use official MLflow documentation navigation rather than guessing a URL.

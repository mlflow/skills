---
name: searching-mlflow-docs
description: Find official MLflow documentation and API examples for GenAI or classic ML. Use for documentation questions, integration guidance, and version-specific API lookup.
---

# Find MLflow documentation

1. Fetch a known exact official topic page, or find it in
   `https://mlflow.org/docs/latest/llms.txt`. The current index includes **GenAI and classic
   ML**; use official ML navigation if a classic-ML topic is absent. Never guess a path.
2. Fetch the relevant Markdown (`.md`) page from `mlflow.org/docs`, requesting complete
   code blocks verbatim. Read only the sections needed for the question. Use `/latest/`
   unless the user requests a specific version; check examples against the installed
   MLflow version and signatures before proposing runnable integration code.
3. Cite the page and return the relevant complete example. Distinguish verbatim source
   from adaptations, and explain any required configuration. Do not present invented,
   summarized or incomplete code as a quoted documentation example.

Prefer official docs over web search or third-party sources. If a page moved, rediscover
it through the current index. If documentation cannot be fetched, say what failed; use
installed help/signatures for a clearly labeled local explanation when sufficient.
Do not fabricate a citation or claim an example was verified or executed when it was not.

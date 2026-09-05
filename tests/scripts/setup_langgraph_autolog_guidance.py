#!/usr/bin/env python3
"""Scaffold a minimal, untraced LangGraph agent for the autolog-guidance fixture.

The agent under test must add MLflow tracing to this app. The judge then checks
that it used a single framework autolog call and did not decorate LangGraph
nodes, tools, or model calls purely to recreate spans autolog already produces.
"""

from __future__ import annotations

import os
from pathlib import Path

AGENT_SOURCE = '''\
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


def lookup_weather(city: str) -> str:
    """A trivial tool the graph can call."""
    return f"It is 72F and sunny in {city}."


def call_model(state: State) -> State:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def call_tool(state: State) -> State:
    reading = lookup_weather("San Francisco")
    return {"messages": [HumanMessage(content=reading)]}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("model", call_model)
    graph.add_node("tool", call_tool)
    graph.set_entry_point("tool")
    graph.add_edge("tool", "model")
    graph.add_edge("model", END)
    return graph.compile()


def run(question: str) -> str:
    app = build_graph()
    result = app.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print(run("What is the weather in San Francisco?"))
'''

REQUIREMENTS = "\n".join(
    [
        "mlflow>=3.8.0",
        "langgraph",
        "langchain-openai",
        "langchain-core",
    ]
)


def main() -> None:
    project_dir = Path(os.environ["PROJECT_DIR"])
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "agent.py").write_text(AGENT_SOURCE, encoding="utf-8")
    (project_dir / "requirements.txt").write_text(REQUIREMENTS + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

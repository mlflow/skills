import json
import os
import subprocess
from pathlib import Path

project = Path(os.environ["PROJECT_DIR"])
project.mkdir(parents=True, exist_ok=True)
if os.environ.get("REPO_URL"):
    subprocess.run(["git", "clone", os.environ["REPO_URL"], str(project)], check=True)
elif os.environ.get("TEST_CASE") == "typescript":
    (project / "agent.ts").write_text('''function normalize(message: string): string {
    return message.toLowerCase();
}
function route_request(message: string): string {
    return normalize(message).includes("refund") ? "support" : "general";
}
console.log(route_request("REFUND"));
console.log(route_request("Hello"));
''')
    (project / "package.json").write_text(json.dumps({"private": True, "type": "module"}))
else:
    (project / "agent.py").write_text('''def normalize(message):
    return message.lower()

def route_request(message):
    return "support" if "refund" in normalize(message) else "general"

if __name__ == "__main__":
    print(route_request("REFUND"))
    print(route_request("Hello"))
''')

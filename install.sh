#!/bin/sh
# MLflow Skills installer — https://github.com/mlflow/skills
#
# One-line install:
#   curl -fsSL https://raw.githubusercontent.com/mlflow/skills/main/install.sh | sh
#
# Installs the MLflow skills into your coding agent's skills directory. Needs
# only `curl` (or `wget`) and `tar` — no git, node, uv, or python required.
#
# Options (flags or environment variables):
#   --agent <name>   claude | codex | cursor | gemini | opencode   (MLFLOW_SKILLS_AGENT)
#   --dir <path>     install into this exact directory instead      (MLFLOW_SKILLS_DIR)
#   --project        install into ./<agent>/skills in the cwd       (MLFLOW_SKILLS_SCOPE=project)
#   --ref <ref>      branch or tag to install (default: main)       (MLFLOW_SKILLS_REF)
#   --list           list the skills that would be installed, then exit
#   -h, --help       show this help
set -eu

REPO="mlflow/skills"
REF="${MLFLOW_SKILLS_REF:-main}"
AGENT="${MLFLOW_SKILLS_AGENT:-}"
DEST="${MLFLOW_SKILLS_DIR:-}"
SCOPE="${MLFLOW_SKILLS_SCOPE:-user}"
LIST_ONLY=0

log()  { printf '%s\n' "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

usage() {
  # Print the leading comment block (lines 2..first non-comment), stripping "# ".
  awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next} {exit}' "$0"
  exit "${1:-0}"
}

# --- parse args ---------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --agent)     AGENT="${2:?--agent needs a value}"; shift 2 ;;
    --agent=*)   AGENT="${1#*=}"; shift ;;
    --dir)       DEST="${2:?--dir needs a value}"; shift 2 ;;
    --dir=*)     DEST="${1#*=}"; shift ;;
    --ref)       REF="${2:?--ref needs a value}"; shift 2 ;;
    --ref=*)     REF="${1#*=}"; shift ;;
    --project)   SCOPE="project"; shift ;;
    --list)      LIST_ONLY=1; shift ;;
    -h|--help)   usage 0 ;;
    *)           die "unknown option: $1 (see --help)" ;;
  esac
done

# --- prerequisites ------------------------------------------------------------
have tar || die "tar is required but not found"
if   have curl; then dl() { curl -fsSL "$1" -o "$2"; }
elif have wget; then dl() { wget -qO "$2" "$1"; }
else die "curl or wget is required but neither was found"; fi

# --- resolve the target skills directory -------------------------------------
dir_for() {
  case "$1" in
    claude)   echo "$HOME/.claude/skills" ;;
    codex)    echo "$HOME/.codex/skills" ;;
    cursor)   echo "$HOME/.cursor/skills" ;;
    gemini)   echo "$HOME/.gemini/skills" ;;
    opencode) echo "$HOME/.config/opencode/skills" ;;
    *)        die "unknown agent: $1 (expected claude|codex|cursor|gemini|opencode)" ;;
  esac
}

if [ -z "$DEST" ]; then
  if [ -z "$AGENT" ]; then
    # Auto-detect: pick the first agent whose home directory already exists.
    for a in claude codex cursor gemini opencode; do
      if [ -d "$(dirname "$(dir_for "$a")")" ]; then AGENT="$a"; break; fi
    done
    AGENT="${AGENT:-claude}"
  fi
  if [ "$SCOPE" = "project" ]; then
    DEST="./.${AGENT}/skills"
  else
    DEST="$(dir_for "$AGENT")"
  fi
fi

# --- download + extract -------------------------------------------------------
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT INT TERM

log "Downloading mlflow/skills@${REF} ..."
dl "https://codeload.github.com/${REPO}/tar.gz/refs/heads/${REF}" "$tmp/skills.tgz" 2>/dev/null \
  || dl "https://codeload.github.com/${REPO}/tar.gz/refs/tags/${REF}" "$tmp/skills.tgz" \
  || die "could not download ref '${REF}' from ${REPO}"
tar -xzf "$tmp/skills.tgz" -C "$tmp"
SRC="$(find "$tmp" -maxdepth 1 -type d -name 'skills-*' | head -1)"
[ -n "$SRC" ] || die "unexpected archive layout"

# --- list mode ----------------------------------------------------------------
if [ "$LIST_ONLY" -eq 1 ]; then
  log "Skills in mlflow/skills@${REF}:"
  for s in "$SRC"/*/; do
    [ -f "${s}SKILL.md" ] && log "  - $(basename "$s")"
  done
  exit 0
fi

# --- copy skills --------------------------------------------------------------
mkdir -p "$DEST"
count=0
for s in "$SRC"/*/; do
  [ -f "${s}SKILL.md" ] || continue        # only real skills, skip tests/hooks/assets
  name="$(basename "$s")"
  rm -rf "$DEST/$name"
  cp -R "$s" "$DEST/$name"
  count=$((count + 1))
done
[ "$count" -gt 0 ] || die "no skills found in archive"
log "Installed $count MLflow skills into $DEST"

# --- inform about the optional auto-suggest hook -----------------------------
# Enabling the hook means editing settings.json. Rather than parse+rewrite JSON
# in shell (which would pull in python/jq and break the curl-only promise), we
# print the steps: a coding agent running this can apply them itself, and a
# human can copy-paste them.
RAW="https://raw.githubusercontent.com/${REPO}/${REF}"
if [ "${AGENT:-claude}" = "claude" ]; then
  log ""
  log "Optional — auto-suggest hook (Claude Code):"
  log "  A hook can surface the right skill automatically from your prompt. To enable it:"
  log "    1. mkdir -p ~/.claude/hooks && \\"
  log "       curl -fsSL ${RAW}/hooks/mlflow-suggest-hook.py -o ~/.claude/hooks/mlflow-suggest-hook.py"
  log "    2. Merge into ~/.claude/settings.json (the hook runs via python3):"
  log '         {"hooks":{"UserPromptSubmit":[{"hooks":[{"type":"command",'
  log '           "command":"python3 ~/.claude/hooks/mlflow-suggest-hook.py"}]}]}}'
  log "  If a coding agent is running this installer, it can make both changes now."
else
  log ""
  log "Optional — an auto-suggest hook is available. Hook setup varies by agent; see:"
  log "  https://github.com/${REPO}/blob/${REF}/hooks/README.md"
fi

# --- done ---------------------------------------------------------------------
log ""
log "Done. Restart your coding agent (or start a new session) to load the skills."

#!/bin/bash
#
# Test script for the agent-evaluation skill
#
# This script:
# 1. Clones a fresh checkout of mlflow-agent
# 2. Starts a local MLflow tracking server (SQLite backend)
# 3. Installs the agent-evaluation skill in .claude/skills/
# 4. Tests Claude Code headless mode with a simple query
# 5. Runs Claude Code evaluation (skill should be auto-discovered)
# 6. Verifies that evaluation artifacts were created
# 7. Copies Claude session logs for inspection
# 8. Cleans up (stops server, removes temp files unless KEEP_WORKDIR)
#
# Usage:
#   ./tests/test_agent_evaluation.sh [extra_prompt]
#
# Arguments:
#   extra_prompt    - Optional text to append to the Claude Code prompt
#
# Options via environment variables:
#   SKILL_DIR            - Path to agent-evaluation skill (default: script's parent dir)
#   TIMEOUT_SECONDS      - Claude Code execution timeout (default: 900 = 15 mins)
#   MLFLOW_PORT          - Port for local MLflow server (default: 5000)
#   KEEP_WORKDIR         - Set to "true" to keep the working directory after completion
#   MLFLOW_TRACKING_URI  - If set, use this external MLflow server instead of starting a local one
#   OPENAI_API_KEY       - If set, passed to Claude Code for MLflow's default judge (OpenAI-based scorers)
#

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SKILL_DIR="${SKILL_DIR:-${REPO_ROOT}/agent-evaluation}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
KEEP_WORKDIR="${KEEP_WORKDIR:-true}"

MLFLOW_AGENT_REPO="https://github.com/alkispoly-db/mlflow-agent"
WORK_DIR=""
TEST_EXPERIMENT_NAME=""
TEST_EXPERIMENT_ID=""
LOG_FILE=""
MLFLOW_SERVER_PID=""
EXTRA_PROMPT="${1:-}"  # Optional extra prompt from first argument
EXTERNAL_MLFLOW_URI="${MLFLOW_TRACKING_URI:-}"  # Optional external MLflow URI
EXTERNAL_OPENAI_KEY="${OPENAI_API_KEY:-}"  # Optional OpenAI API key for MLflow judges
USE_EXTERNAL_SERVER=false

# Exit codes
EXIT_SUCCESS=0
EXIT_SETUP_FAILED=1
EXIT_EXECUTION_FAILED=2
EXIT_VERIFICATION_FAILED=3

# ============================================================================
# Logging utilities
# ============================================================================

log_info() {
    echo "[INFO] $*"
}

log_error() {
    echo "[ERROR] $*" >&2
}

log_success() {
    echo "[SUCCESS] $*"
}

log_section() {
    echo ""
    echo "========================================"
    echo "$*"
    echo "========================================"
}

# ============================================================================
# Cleanup
# ============================================================================

cleanup() {
    local exit_code=$?

    log_section "Cleanup"

    # Copy Claude session logs to work directory for inspection
    if [[ -n "${WORK_DIR}" && -d "${WORK_DIR}" ]]; then
        local claude_projects_dir="${HOME}/.claude/projects"
        local project_path_encoded=$(echo "${WORK_DIR}/mlflow-agent" | sed 's|/|-|g')
        local session_dir="${claude_projects_dir}/${project_path_encoded}"

        if [[ -d "${session_dir}" ]]; then
            log_info "Copying Claude session logs..."
            mkdir -p "${WORK_DIR}/claude-sessions"
            cp -r "${session_dir}"/* "${WORK_DIR}/claude-sessions/" 2>/dev/null || true
            log_info "Claude session logs copied to: ${WORK_DIR}/claude-sessions/"
        fi
    fi

    # Stop MLflow server if we started one (not when using external server)
    if [[ -n "${MLFLOW_SERVER_PID}" && "${USE_EXTERNAL_SERVER}" != "true" ]]; then
        log_info "Stopping MLflow server (PID: ${MLFLOW_SERVER_PID})..."
        kill "${MLFLOW_SERVER_PID}" 2>/dev/null || true
        wait "${MLFLOW_SERVER_PID}" 2>/dev/null || true
        log_info "MLflow server stopped"
    fi

    # Remove working directory unless KEEP_WORKDIR is true
    if [[ -n "${WORK_DIR}" && -d "${WORK_DIR}" && "${KEEP_WORKDIR}" != "true" ]]; then
        log_info "Removing working directory: ${WORK_DIR}"
        rm -rf "${WORK_DIR}"
    elif [[ -n "${WORK_DIR}" && -d "${WORK_DIR}" ]]; then
        log_info "Keeping working directory: ${WORK_DIR}"
        log_info "  Claude Code output log: ${LOG_FILE:-N/A}"
        log_info "  Claude session logs: ${WORK_DIR}/claude-sessions/"
        if [[ "${USE_EXTERNAL_SERVER}" == "true" ]]; then
            log_info "  MLflow tracking URI: ${EXTERNAL_MLFLOW_URI}"
        else
            log_info "  MLflow data: ${WORK_DIR}/mlflow-data"
        fi
        log_info "  Evaluation experiment ID: ${TEST_EXPERIMENT_ID:-N/A}"
    fi

    exit ${exit_code}
}

trap cleanup EXIT

# ============================================================================
# Prerequisites validation
# ============================================================================

check_prerequisites() {
    log_section "Checking Prerequisites"

    local missing_deps=()

    # Check required commands
    if ! command -v claude &>/dev/null; then
        missing_deps+=("claude (Claude Code CLI)")
    fi

    if ! command -v git &>/dev/null; then
        missing_deps+=("git")
    fi

    if ! command -v uv &>/dev/null; then
        missing_deps+=("uv (Python package manager)")
    fi

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing required commands:"
        for dep in "${missing_deps[@]}"; do
            log_error "  - ${dep}"
        done
        return 1
    fi

    log_info "All required commands available: claude, git, uv"

    # Check skill directory exists
    if [[ ! -d "${SKILL_DIR}" ]]; then
        log_error "Skill directory not found: ${SKILL_DIR}"
        return 1
    fi

    if [[ ! -f "${SKILL_DIR}/SKILL.md" ]]; then
        log_error "SKILL.md not found in: ${SKILL_DIR}"
        return 1
    fi

    log_info "Skill directory found: ${SKILL_DIR}"

    # Check if using external MLflow server
    if [[ -n "${EXTERNAL_MLFLOW_URI}" ]]; then
        USE_EXTERNAL_SERVER=true
        log_info "Using external MLflow server: ${EXTERNAL_MLFLOW_URI}"
    else
        # Check if port is available (only needed for local server)
        if lsof -i ":${MLFLOW_PORT}" &>/dev/null; then
            log_error "Port ${MLFLOW_PORT} is already in use"
            log_error "Set MLFLOW_PORT to use a different port"
            return 1
        fi
        log_info "Port ${MLFLOW_PORT} is available"
    fi

    log_success "All prerequisites satisfied"
    return 0
}

# ============================================================================
# Start local MLflow server
# ============================================================================

start_mlflow_server() {
    log_section "Starting Local MLflow Server"

    # Create directories for MLflow data
    mkdir -p "${WORK_DIR}/mlflow-data/artifacts"

    local backend_store="sqlite:///${WORK_DIR}/mlflow-data/mlflow.db"
    local artifact_root="${WORK_DIR}/mlflow-data/artifacts"

    log_info "Backend store: ${backend_store}"
    log_info "Artifact root: ${artifact_root}"
    log_info "Starting server on port ${MLFLOW_PORT}..."

    # Start MLflow server in background (must be run from project dir where uv sync was done)
    uv run python -m mlflow server \
        --host 127.0.0.1 \
        --port "${MLFLOW_PORT}" \
        --backend-store-uri "${backend_store}" \
        --default-artifact-root "${artifact_root}" \
        > "${WORK_DIR}/mlflow-server.log" 2>&1 &

    MLFLOW_SERVER_PID=$!

    # Wait for server to be ready
    local max_attempts=30
    local attempt=0
    while [[ ${attempt} -lt ${max_attempts} ]]; do
        if curl -s "http://127.0.0.1:${MLFLOW_PORT}/health" &>/dev/null; then
            break
        fi
        # Check if process died
        if ! kill -0 "${MLFLOW_SERVER_PID}" 2>/dev/null; then
            log_error "MLflow server process died"
            log_error "Check log: ${WORK_DIR}/mlflow-server.log"
            cat "${WORK_DIR}/mlflow-server.log" >&2
            return 1
        fi
        sleep 1
        ((attempt++))
    done

    if [[ ${attempt} -ge ${max_attempts} ]]; then
        log_error "MLflow server failed to start (timeout)"
        log_error "Check log: ${WORK_DIR}/mlflow-server.log"
        cat "${WORK_DIR}/mlflow-server.log" >&2
        return 1
    fi

    log_info "MLflow server started (PID: ${MLFLOW_SERVER_PID})"

    # Set environment variables for MLflow
    export MLFLOW_TRACKING_URI="http://127.0.0.1:${MLFLOW_PORT}"
    log_info "MLFLOW_TRACKING_URI set to: ${MLFLOW_TRACKING_URI}"

    log_success "MLflow server is ready"
    return 0
}

# ============================================================================
# Setup phase
# ============================================================================

setup_phase() {
    log_section "Setup Phase"

    # Create working directory (in repo's test-runs folder for easier access)
    local test_runs_dir="${REPO_ROOT}/test-runs"
    mkdir -p "${test_runs_dir}"
    WORK_DIR=$(mktemp -d "${test_runs_dir}/agent-eval-test-XXXXXX")
    log_info "Created working directory: ${WORK_DIR}"

    # Clone mlflow-agent repository
    log_info "Cloning mlflow-agent repository..."
    if ! git clone --depth 1 "${MLFLOW_AGENT_REPO}" "${WORK_DIR}/mlflow-agent" 2>&1; then
        log_error "Failed to clone repository"
        return 1
    fi
    log_info "Repository cloned successfully"

    # Copy skill to .claude/skills directory (standard skill location)
    log_info "Setting up agent-evaluation skill in .claude/skills/..."
    mkdir -p "${WORK_DIR}/mlflow-agent/.claude/skills"
    cp -r "${SKILL_DIR}" "${WORK_DIR}/mlflow-agent/.claude/skills/agent-evaluation"
    log_info "Skill installed at: ${WORK_DIR}/mlflow-agent/.claude/skills/agent-evaluation"

    # Change to mlflow-agent directory for uv commands
    pushd "${WORK_DIR}/mlflow-agent" > /dev/null

    # Install dependencies first
    log_info "Installing Python dependencies with uv sync..."
    if ! uv sync 2>&1; then
        log_error "Failed to install dependencies"
        popd > /dev/null
        return 1
    fi
    log_info "Dependencies installed successfully"

    # Add mlflow package (needed for local server and evaluation)
    log_info "Adding mlflow package..."
    if ! uv add mlflow 2>&1; then
        log_error "Failed to add mlflow package"
        popd > /dev/null
        return 1
    fi
    log_info "MLflow package added"

    # Start local MLflow server or use external one
    if [[ "${USE_EXTERNAL_SERVER}" == "true" ]]; then
        log_section "Using External MLflow Server"
        export MLFLOW_TRACKING_URI="${EXTERNAL_MLFLOW_URI}"
        log_info "MLFLOW_TRACKING_URI set to: ${MLFLOW_TRACKING_URI}"

        # Install Databricks-specific packages if using Databricks tracking URI
        if [[ "${EXTERNAL_MLFLOW_URI}" == databricks://* ]]; then
            log_info "Installing Databricks packages for Unity Catalog dataset support..."
            if ! uv add databricks-agents databricks-connect 2>&1; then
                log_error "Failed to install Databricks packages"
                log_error "Dataset operations may fail without databricks-agents and databricks-connect"
                # Don't fail - evaluation can still work with local DataFrames
            else
                log_info "Databricks packages installed (databricks-agents, databricks-connect)"
            fi
        fi

        log_success "External MLflow server configured"
    else
        # Start local MLflow server (must be after uv sync so mlflow is available)
        if ! start_mlflow_server; then
            popd > /dev/null
            return 1
        fi
    fi

    # Create test experiment
    local timestamp=$(date +%Y%m%d-%H%M%S)
    local base_experiment_name="agent-eval-test-${timestamp}"

    # For Databricks, experiment names must be absolute workspace paths
    if [[ "${USE_EXTERNAL_SERVER}" == "true" && "${EXTERNAL_MLFLOW_URI}" == databricks://* ]]; then
        # Extract profile name from databricks://<profile> URI
        local db_profile="${EXTERNAL_MLFLOW_URI#databricks://}"
        log_info "Detecting Databricks workspace user for profile: ${db_profile}"

        # Get current user's email from Databricks CLI
        local db_user
        db_user=$(databricks current-user me -p "${db_profile}" 2>/dev/null | python3 -c "import sys, json; print(json.load(sys.stdin).get('userName', ''))" 2>/dev/null || echo "")

        if [[ -z "${db_user}" ]]; then
            log_error "Failed to get Databricks user. Make sure 'databricks auth login -p ${db_profile}' has been run."
            popd > /dev/null
            return 1
        fi

        TEST_EXPERIMENT_NAME="/Users/${db_user}/${base_experiment_name}"
        log_info "Using Databricks workspace path for experiment"
    else
        TEST_EXPERIMENT_NAME="${base_experiment_name}"
    fi

    log_info "Creating test experiment: ${TEST_EXPERIMENT_NAME}"

    # Create experiment for agent evaluation and capture its ID
    TEST_EXPERIMENT_ID=$(uv run python -c "
import mlflow
experiment_id = mlflow.create_experiment('${TEST_EXPERIMENT_NAME}')
print(experiment_id)
" 2>&1)

    if [[ -z "${TEST_EXPERIMENT_ID}" || "${TEST_EXPERIMENT_ID}" == *"Error"* || "${TEST_EXPERIMENT_ID}" == *"Traceback"* ]]; then
        log_error "Failed to create test experiment: ${TEST_EXPERIMENT_ID}"
        popd > /dev/null
        return 1
    fi

    log_info "Created evaluation experiment with ID: ${TEST_EXPERIMENT_ID}"

    # Export the experiment ID for the evaluation task
    export MLFLOW_EXPERIMENT_ID="${TEST_EXPERIMENT_ID}"

    # Export OpenAI API key if provided (for MLflow's default judge)
    if [[ -n "${EXTERNAL_OPENAI_KEY}" ]]; then
        export OPENAI_API_KEY="${EXTERNAL_OPENAI_KEY}"
        log_info "OPENAI_API_KEY set (for MLflow scorers)"
    fi

    popd > /dev/null

    log_success "Setup phase completed"
    return 0
}

# ============================================================================
# Test Claude Code headless mode
# ============================================================================

test_claude_headless() {
    log_section "Testing Claude Code Headless Mode"

    cd "${WORK_DIR}/mlflow-agent"

    log_info "Running simple Claude Code test query..."
    local claude_test_output
    claude_test_output=$(timeout 120 claude -p "Say hello world" --allowedTools "" < /dev/null 2>&1) || true

    if [[ -z "${claude_test_output}" ]]; then
        log_error "Claude Code headless mode test failed - no output"
        return 1
    fi

    if [[ "${claude_test_output}" == *"error"* || "${claude_test_output}" == *"Error"* ]]; then
        log_error "Claude Code headless mode test failed"
        log_error "Output: ${claude_test_output}"
        return 1
    fi

    log_info "Claude Code responded: ${claude_test_output:0:100}..."
    log_success "Claude Code headless mode is working"
    return 0
}

# ============================================================================
# Run Claude Code
# ============================================================================

run_claude_code() {
    log_section "Running Claude Code Evaluation"

    LOG_FILE="${WORK_DIR}/claude_output.log"

    log_info "Changing to mlflow-agent directory..."
    cd "${WORK_DIR}/mlflow-agent"

    # Build the prompt
    local base_prompt="Evaluate the output quality of my agent. Do not ask for input."
    local full_prompt="${base_prompt}"
    if [[ -n "${EXTRA_PROMPT}" ]]; then
        full_prompt="${base_prompt} ${EXTRA_PROMPT}"
        log_info "Extra prompt: ${EXTRA_PROMPT}"
    fi

    log_info "Executing Claude Code with agent-evaluation skill..."
    log_info "Prompt: ${full_prompt}"
    log_info "Timeout: ${TIMEOUT_SECONDS} seconds"
    log_info "Log file: ${LOG_FILE}"

    # Run Claude Code - skill should be auto-discovered from .claude/skills/
    local exit_code=0
    timeout "${TIMEOUT_SECONDS}" claude \
        -p "${full_prompt}" \
        --dangerously-skip-permissions \
        --allowedTools "Bash,Read,Write,Edit,Grep,Glob,WebFetch" \
        < /dev/null > "${LOG_FILE}" 2>&1 || exit_code=$?

    if [[ ${exit_code} -eq 124 ]]; then
        log_info "Claude Code execution timed out after ${TIMEOUT_SECONDS} seconds"
        log_info "Will still verify if artifacts were created before timeout"
        # Don't return - continue to verification
    elif [[ ${exit_code} -ne 0 ]]; then
        log_error "Claude Code exited with code: ${exit_code}"
        log_error "Check log file for details: ${LOG_FILE}"
        return 1
    else
        log_success "Claude Code execution completed"
    fi

    return 0
}

# ============================================================================
# Verification phase
# ============================================================================

verify_results() {
    log_section "Verification Phase"

    cd "${WORK_DIR}/mlflow-agent"

    # Run Python verification script
    log_info "Running verification checks..."

    local verification_result
    verification_result=$(uv run python -c "
import sys
import json
from mlflow import MlflowClient

experiment_id = '${TEST_EXPERIMENT_ID}'
client = MlflowClient()

results = {
    'datasets': {'found': 0, 'pass': False},
    'scorers': {'found': 0, 'pass': False},
    'traces': {'found': 0, 'with_assessments': 0, 'pass': False}
}

# Check datasets
try:
    datasets = client.search_datasets(experiment_ids=[experiment_id])
    results['datasets']['found'] = len(datasets)
    results['datasets']['pass'] = len(datasets) >= 1
except Exception as e:
    print(f'Warning: Error checking datasets: {e}', file=sys.stderr)

# Check traces with assessments (evaluation runs)
try:
    traces = client.search_traces(experiment_ids=[experiment_id])
    results['traces']['found'] = len(traces)

    traces_with_assessments = 0
    for trace in traces:
        # Check if trace has assessments (from evaluation)
        trace_info = client.get_trace(trace.info.request_id)
        if hasattr(trace_info, 'info') and hasattr(trace_info.info, 'assessments'):
            if trace_info.info.assessments:
                traces_with_assessments += 1

    results['traces']['with_assessments'] = traces_with_assessments
    # Pass if we have at least 1 trace (evaluation was attempted)
    results['traces']['pass'] = len(traces) >= 1
except Exception as e:
    print(f'Warning: Error checking traces: {e}', file=sys.stderr)

# Check scorers (via experiment tags or registered scorers)
try:
    import subprocess
    scorer_result = subprocess.run(
        ['uv', 'run', 'python', '-m', 'mlflow', 'scorers', 'list', '-x', experiment_id],
        capture_output=True,
        text=True
    )
    # Count non-empty lines that aren't headers
    lines = [l for l in scorer_result.stdout.strip().split('\n') if l and not l.startswith('---') and 'Name' not in l]
    results['scorers']['found'] = len(lines)
    results['scorers']['pass'] = len(lines) >= 1
except Exception as e:
    print(f'Warning: Error checking scorers: {e}', file=sys.stderr)

print(json.dumps(results))
" 2>/dev/null)

    if [[ -z "${verification_result}" ]]; then
        log_error "Verification script failed to produce output"
        return 1
    fi

    # Parse and display results
    log_info "Verification Results:"
    echo ""

    local datasets_found datasets_pass scorers_found scorers_pass traces_found traces_pass
    datasets_found=$(echo "${verification_result}" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['datasets']['found'])")
    datasets_pass=$(echo "${verification_result}" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['datasets']['pass'])")
    scorers_found=$(echo "${verification_result}" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['scorers']['found'])")
    scorers_pass=$(echo "${verification_result}" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['scorers']['pass'])")
    traces_found=$(echo "${verification_result}" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['traces']['found'])")
    traces_pass=$(echo "${verification_result}" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['traces']['pass'])")

    local all_pass=true

    # Datasets check
    if [[ "${datasets_pass}" == "True" ]]; then
        echo "  [PASS] Datasets: ${datasets_found} created"
    else
        echo "  [FAIL] Datasets: ${datasets_found} created (expected >= 1)"
        all_pass=false
    fi

    # Scorers check
    if [[ "${scorers_pass}" == "True" ]]; then
        echo "  [PASS] Scorers: ${scorers_found} registered"
    else
        echo "  [FAIL] Scorers: ${scorers_found} registered (expected >= 1)"
        all_pass=false
    fi

    # Traces/Evaluation runs check
    if [[ "${traces_pass}" == "True" ]]; then
        echo "  [PASS] Evaluation runs: ${traces_found} traces created"
    else
        echo "  [FAIL] Evaluation runs: ${traces_found} traces created (expected >= 1)"
        all_pass=false
    fi

    echo ""

    if [[ "${all_pass}" == "true" ]]; then
        log_success "All verification checks passed"
        return 0
    else
        log_error "Some verification checks failed"
        return 1
    fi
}

# ============================================================================
# Main
# ============================================================================

main() {
    log_section "Agent Evaluation Skill Test"
    log_info "Starting test at $(date)"
    if [[ -n "${EXTRA_PROMPT}" ]]; then
        log_info "Extra prompt argument: ${EXTRA_PROMPT}"
    fi
    if [[ -n "${EXTERNAL_MLFLOW_URI}" ]]; then
        log_info "External MLflow URI: ${EXTERNAL_MLFLOW_URI}"
    fi
    if [[ -n "${EXTERNAL_OPENAI_KEY}" ]]; then
        log_info "OPENAI_API_KEY: provided (for MLflow scorers)"
    fi

    # Phase 1: Check prerequisites
    if ! check_prerequisites; then
        log_error "Prerequisites check failed"
        exit ${EXIT_SETUP_FAILED}
    fi

    # Phase 2: Setup (includes starting MLflow server)
    if ! setup_phase; then
        log_error "Setup phase failed"
        exit ${EXIT_SETUP_FAILED}
    fi

    # Phase 3: Test Claude Code headless mode
    if ! test_claude_headless; then
        log_error "Claude Code headless mode test failed"
        exit ${EXIT_SETUP_FAILED}
    fi

    # Phase 4: Run Claude Code evaluation
    if ! run_claude_code; then
        log_error "Claude Code execution failed"
        log_error "Check log file: ${LOG_FILE}"
        exit ${EXIT_EXECUTION_FAILED}
    fi

    # Phase 5: Verify results
    if ! verify_results; then
        log_error "Verification failed"
        exit ${EXIT_VERIFICATION_FAILED}
    fi

    log_section "Test Completed Successfully"
    log_info "Experiment: ${TEST_EXPERIMENT_NAME} (ID: ${TEST_EXPERIMENT_ID})"
    log_info "All evaluation artifacts were created as expected"

    exit ${EXIT_SUCCESS}
}

# Run main
main "$@"

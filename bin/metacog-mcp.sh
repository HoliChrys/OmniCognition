#!/usr/bin/env bash
# Launcher for the metacog MCP server as a Claude Code plugin.
#
# Resolves the brain (storage) the same way the hooks do :
#   1. a `.metacog-brain` marker walked up from the current project directory
#      (first non-empty, non-comment line = the storage path) — a dev repo keeps
#      its OWN memory instead of polluting the shared one ;
#   2. $METACOG_STORAGE ;
#   3. ~/.metacog/memory.pkl (shared default).
# Exposes the agent-facing `external` surface unless METACOG_SURFACE says
# otherwise, and imports `metacog` from the plugin root when not pip-installed.
set -euo pipefail

ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

brain=""
d="$PWD"
while [ -n "$d" ] && [ "$d" != "/" ]; do
  if [ -f "$d/.metacog-brain" ]; then
    brain="$(grep -v '^[[:space:]]*#' "$d/.metacog-brain" | grep -v '^[[:space:]]*$' | head -n 1 || true)"
    break
  fi
  d="$(dirname "$d")"
done
STORAGE="${brain:-${METACOG_STORAGE:-$HOME/.metacog/memory.pkl}}"
STORAGE="${STORAGE/#\~/$HOME}"

export METACOG_SURFACE="${METACOG_SURFACE:-external}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
exec "${METACOG_PYTHON:-python3}" -m metacog.mcp_server --storage "$STORAGE" "$@"

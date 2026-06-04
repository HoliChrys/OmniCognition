"""
Sandboxed Python executor for tool-tagged ACTION points.

A "tool" is an ACTION Point with a non-None `exec_spec` AND the "tool"
tag — both required, both opt-in. Discovery is semantic (any retrieve
that surfaces a tool-tagged Point can suggest executing it) ; execution
is delegated to PyExecutor which rolls the code in an isolated Python
subprocess and parses the JSON result.

The PROCESS of execution is COMPUTATION (deterministic, isolated, no
network of its own design) — Cor.5 compliant. The resulting FACT child
inherits legitimacy from its parent ACTION-tool ; it is an observation
of a deterministic computation, not GENERATOR output.

Convention for the embedded code :
    def run(args: dict) -> JSON-serializable
The wrapper around `code` reads args from stdin, calls run(args), and
prints the JSON-encoded result on the LAST line of stdout. PyExecutor
parses that last line ; any prints by `run` itself before the final
result are tolerated and ignored.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional

from metacog.epistemic import SourceClass


# Audit : the execution process itself is COMPUTATION (deterministic
# isolated subprocess, no LLM in the loop).
EXECUTOR_SOURCE: SourceClass = SourceClass.COMPUTATION

# Wrapper that runs around the user's code. Reads args from stdin (JSON),
# calls run(args), prints json.dumps(result) on the LAST line of stdout.
_WRAPPER = (
    "import json, sys\n"
    "{code}\n"
    "_args = json.loads(sys.stdin.read() or 'null')\n"
    "_result = run(_args)\n"
    "print('\\n__RESULT__' + json.dumps(_result))\n"
)


class PyExecutor:
    """Subprocess-isolated runner for ACTION exec_specs."""

    def __init__(self, timeout: int = 5) -> None:
        self.timeout = timeout

    def execute(self, exec_spec: Optional[Dict[str, Any]],
                args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Roll `exec_spec` with `args` and return {ok, result|error}.

        Returns
        -------
        {"ok": True,  "result": <JSON value>}      on success
        {"ok": False, "error":  "<short message>"}  on any failure
        """
        if not exec_spec or not isinstance(exec_spec, dict):
            return {"ok": False, "error": "exec_spec is missing or invalid"}
        if exec_spec.get("lang", "python") != "python":
            return {"ok": False, "error": f"unsupported lang {exec_spec.get('lang')!r}"}
        code = exec_spec.get("code")
        if not isinstance(code, str) or "def run" not in code:
            return {"ok": False, "error": "exec_spec.code must define def run(args)"}
        payload = args if args is not None else {}
        if not isinstance(payload, dict):
            return {"ok": False, "error": "args must be a dict"}

        script = _WRAPPER.format(code=code)
        try:
            # Write to a temp file so tracebacks point at a real path ;
            # python -I = isolated mode (ignore PYTHONPATH + user site).
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as fh:
                fh.write(script)
                script_path = fh.name
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", script_path],
                    input=json.dumps(payload),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env={"LANG": "C"},
                    shell=False,
                )
            finally:
                try:
                    os.unlink(script_path)
                except OSError:
                    pass
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout after {self.timeout}s"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"subprocess error: {exc}"}

        if proc.returncode != 0:
            err = (proc.stderr or "").strip().splitlines()
            tail = err[-1] if err else f"exit {proc.returncode}"
            return {"ok": False, "error": tail[:200]}

        # The result is on the line that starts with __RESULT__. Anything
        # the user code printed before is tolerated and ignored.
        marker = "__RESULT__"
        for line in (proc.stdout or "").splitlines()[::-1]:
            if line.startswith(marker):
                try:
                    return {"ok": True, "result": json.loads(line[len(marker):])}
                except json.JSONDecodeError as exc:
                    return {"ok": False, "error": f"non-JSON output: {exc}"}
        return {"ok": False,
                "error": f"no result marker in stdout: {(proc.stdout or '')[:100]!r}"}

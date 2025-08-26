from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Any
import shlex
import os.path as _op

class EnvManager:
    """
    Simple multi-script process manager.
    - Reads a JSON config: {"name": "command string", ...}
    - run(name): starts process in background, spawns reader threads for stdout/stderr
    - stop(name): SIGTERM -> wait -> SIGKILL fallback
    - get_status(name): returns {"name", "status", "pid"}
    - get_logs(name): returns captured lines: "[STDOUT] ...", "[STDERR] ..."]
    - stop_all(): stops everything safely (used by pytest teardown)
    """

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.scripts: Dict[str, str] = cfg.get("scripts", {})

        # processes[name] = {"proc": Popen, "threads": [t_out, t_err]}
        self.processes: Dict[str, Dict[str, Any]] = {}

        # output storage per process
        self.process_outputs: Dict[str, List[str]] = defaultdict(list)

        # last stop signal per process (for diagnostics if needed)
        self._last_stop_signal: Dict[str, Optional[int]] = defaultdict(lambda: None)

    # ---------- Internal helpers ----------

    def _read_output_stream(self, process_name: str, stream, stream_type: str) -> None:
        """
        Read a text stream line-by-line and append to self.process_outputs.
        Assumes Popen(..., text=True).
        """
        try:
            for line in iter(stream.readline, ''):
                line = line.rstrip('\r\n')
                if not line:
                    continue
                self.process_outputs.setdefault(process_name, []).append(
                    f"[{stream_type.upper()}] {line}"
                )
        except Exception:
            # Avoid crashing tests due to background thread exceptions
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _get_entry(self, name: str) -> Optional[Dict[str, Any]]:
        return self.processes.get(name)

    def _get_proc(self, name: str) -> Optional[subprocess.Popen]:
        entry = self._get_entry(name)
        if not entry: # Check if entry exists before accessing 'proc'
            return None
        proc = entry.get("proc")
        return proc if isinstance(proc, subprocess.Popen) else None

    # ---------- Public API ----------

    def run(self, script_name: str) -> None:
        cmd = self.scripts.get(script_name)
        if not cmd:
            raise ValueError(f"Script '{script_name}' not found in config")

        # If already running, do nothing
        existing = self._get_proc(script_name)
        if existing and existing.poll() is None:
            return

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        def _spawn_shell(c: str) -> subprocess.Popen:
            # If it's a python one-liner, force unbuffered
            c2 = c
            if c2.strip().startswith("python") and " -u " not in c2 and " -c " in c2:
                c2 = c2.replace("python", "python -u", 1)
            return subprocess.Popen(
                c2, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                bufsize=1, shell=True, env=env,
            )

        if isinstance(cmd, str):
            # Prefer shell-free for safety/speed…
            try:
                argv = shlex.split(cmd)
                if argv and _op.basename(argv[0]) in ("python","python3","python3.13","python3.12","python3.11") and "-u" not in argv:
                    argv.insert(1, "-u")
                try:
                    proc = subprocess.Popen(
                        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, encoding="utf-8", errors="replace",
                        bufsize=1, shell=False, env=env,
                    )
                except FileNotFoundError:
                    # 👈 Fallback when executable does not exist, e.g. "this_command_does_not_exist"
                    proc = _spawn_shell(cmd)
            except ValueError:
                # Complex quoting (e.g., python -c "..."): use shell
                proc = _spawn_shell(cmd)
        else:
            argv = list(cmd)
            if argv and _op.basename(argv[0]) in ("python","python3","python3.13","python3.12","python3.11") and "-u" not in argv:
                argv.insert(1, "-u")
            try:
                proc = subprocess.Popen(
                    argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, encoding="utf-8", errors="replace",
                    bufsize=1, shell=False, env=env,
                )
            except FileNotFoundError:
                # Last-resort: join to string and run via shell
                proc = _spawn_shell(" ".join(argv))

        t_out = threading.Thread(target=self._read_output_stream, args=(script_name, proc.stdout, "stdout"), daemon=True)
        t_err = threading.Thread(target=self._read_output_stream, args=(script_name, proc.stderr, "stderr"), daemon=True)
        t_out.start(); t_err.start()

        self.processes[script_name] = {"proc": proc, "threads": [t_out, t_err]}
        self._last_stop_signal[script_name] = None

        # Optional: small delay to let threads attach before tests assert logs/status
        # time.sleep(0.01)

    def stop(self, script_name: str) -> None:
        entry = self._get_entry(script_name)
        proc = self._get_proc(script_name)
        if not entry or not proc:
            return

        # Already exited
        if proc.poll() is not None:
            return

        # Try graceful TERM
        self._last_stop_signal[script_name] = signal.SIGTERM
        try:
            proc.terminate()  # SIGTERM (15)
        except ProcessLookupError:
            pass

        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            # Fallback KILL
            self._last_stop_signal[script_name] = signal.SIGKILL
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            finally:
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass

    def stop_all(self) -> None:
        # Iterate over a static list to avoid dict size change during iteration
        for script_name in list(self.processes.keys()):
            try:
                self.stop(script_name)
            except Exception:
                # Don't let teardown explode on one failure
                pass

    def get_status(self, script_name: str) -> Dict[str, Any]:
        """
        Returns a dict:
        - running            -> {"status": "running", "pid": <pid>}
        - clean exit (0)     -> {"status": "stopped", "pid": None}
        - signal exit (<0)   -> {"status": "terminated", "pid": None}
        - error exit (>0)    -> {"status": "failed (<code>)", "pid": None}
        - not started        -> {"status": "idle", "pid": None}
        """
        proc = self._get_proc(script_name)
        if not proc:
            return {"name": script_name, "status": "idle", "pid": None}

        rc = proc.poll()
        if rc is None:
            return {"name": script_name, "status": "running", "pid": proc.pid}
        if rc == 0:
            return {"name": script_name, "status": "stopped", "pid": None}
        if rc < 0:
            return {"name": script_name, "status": "terminated", "pid": None}
        return {"name": script_name, "status": f"failed ({rc})", "pid": None}

    def get_logs(self, script_name: str) -> List[str]:
        entry = self._get_entry(script_name)
        if entry: # Check if entry exists before accessing 'process_outputs'
            proc = self._get_proc(script_name)
            # If process already exited, give reader threads a short moment to finish last reads
            if proc and proc.poll() is not None:
                for _ in range(2):
                    time.sleep(0.05)
        return list(self.process_outputs.get(script_name, []))

    # Optional utility for diagnostics
    def get_last_stop_signal(self, script_name: str) -> Optional[int]:
        return self._last_stop_signal.get(script_name)
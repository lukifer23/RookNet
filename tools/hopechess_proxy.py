#!/usr/bin/env python3
"""Lightweight proxy that launches HopeChess and filters out noisy debug
prints so that the outer UCI consumer (python-chess) only sees well-formed
protocol lines.

Usage: hopechess_proxy.py /absolute/path/to/HopeChess [--uci args]
The proxy echoes all stdin → child stdin unchanged and only forwards stdout
lines that look like UCI commands (id, option, info, bestmove, etc.). All
stderr from HopeChess is suppressed.
"""
from __future__ import annotations
import subprocess, sys, threading, re, os, signal, select, time

# --- Helpers --------------------------------------------------------------
UCI_RE = re.compile(r"^(id|option|uciok|readyok|bestmove|info|copyprotection|registration|checkready|unknowncommand|go|stop|ponderhit|quit)", re.I)

def start_child(cmd: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        close_fds=os.name != 'nt',
    )

# --- Main -----------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: hopechess_proxy.py /path/to/HopeChess --uci", file=sys.stderr)
        sys.exit(1)

    child_cmd = sys.argv[1:]
    child = start_child(child_cmd)

    # Forward stdin to child in a background thread
    def pump_stdin():
        try:
            for line in sys.stdin:
                if child.poll() is not None:
                    break
                child.stdin.write(line)
                child.stdin.flush()
        except BrokenPipeError:
            pass
        finally:
            try:
                child.stdin.close()
            except Exception:
                pass

    threading.Thread(target=pump_stdin, daemon=True).start()

    try:
        for line in child.stdout:
            if not line:
                break
            if UCI_RE.match(line.lstrip()):
                sys.stdout.write(line)
                sys.stdout.flush()
            # else: drop noisy debug.
    except KeyboardInterrupt:
        pass
    finally:
        try:
            child.terminate()
        except Exception:
            pass
        child.wait(timeout=5)

if __name__ == "__main__":
    main() 
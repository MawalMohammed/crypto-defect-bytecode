"""Runner: gold eval -> rule baselines (both fast) -> CI/seeds (longest). Keep-awake."""
import subprocess
import sys

try:
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
except Exception:
    pass

for script in ["harden_gold.py", "harden_rules.py", "harden_ci.py"]:
    print(f"=== running {script} ===", flush=True)
    r = subprocess.run([sys.executable, "-u", script])
    print(f"=== {script} exit {r.returncode} ===", flush=True)
    if r.returncode:
        sys.exit(r.returncode)
print("=== all done ===", flush=True)

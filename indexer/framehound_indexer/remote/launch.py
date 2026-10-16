# Runs in the Colab kernel (``colab exec -f``) and must return within seconds:
# it starts the indexer as a detached process, which inherits the kernel's CUDA
# environment and survives the exec, then prints a marker and the pid.
# Env: FH_JOB (job.json path), FH_PKG (dir holding framehound_indexer), FH_CMD
# (optional JSON argv prefix, for tests).
import json as _json
import os as _os
import subprocess as _sp
import sys as _sys


def _fh_launch():
    job = _os.environ["FH_JOB"]
    work = _os.path.dirname(job)
    pidfile = _os.path.join(work, "pid")
    if _os.path.exists(pidfile):
        with open(pidfile) as fh:
            pid = int(fh.read().strip() or 0)
        try:
            _os.kill(pid, 0)
            print("@@fh-launched", pid, "already-running")
            return
        except OSError:
            pass
    pkg = _os.environ["FH_PKG"]
    env = dict(_os.environ, PYTHONPATH=pkg + _os.pathsep + _os.environ.get("PYTHONPATH", ""),
               PYTHONUNBUFFERED="1")
    cmd = _json.loads(_os.environ.get("FH_CMD") or "null") or [
        _sys.executable, _os.path.join(pkg, "framehound_indexer", "remote", "run.py")]
    with open(_os.path.join(work, "run.log"), "ab") as log:
        p = _sp.Popen([*cmd, job], env=env, cwd=work, stdin=_sp.DEVNULL, stdout=log,
                      stderr=_sp.STDOUT, start_new_session=True)
    with open(pidfile, "w") as fh:
        fh.write(str(p.pid))
    print("@@fh-launched", p.pid)


_fh_launch()

# Runs in the Colab kernel (``colab exec -f``): prints one marker line with the
# progress lines written since FH_OFFSET, whether the indexer is still alive,
# and (once it has exited) the tail of its log.
import json as _json
import os as _os


def _fh_alive(pid):
    try:
        # The kernel is the parent: reap the child if it exited, or it stays a zombie.
        done, _ = _os.waitpid(pid, _os.WNOHANG)
        if done == pid:
            return False
    except ChildProcessError:
        pass
    except OSError:
        pass
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except OSError:
        pass
    try:
        _os.kill(pid, 0)
        return True
    except OSError:
        return False


def _fh_poll():
    work = _os.environ["FH_WORK"]
    offset = int(_os.environ.get("FH_OFFSET") or 0)
    out = {"lines": [], "offset": offset, "alive": False, "log": None}
    try:
        with open(_os.path.join(work, "pid")) as fh:
            pid = int(fh.read().strip())
        out["alive"] = _fh_alive(pid)
    except (OSError, ValueError):
        pid = None
    path = _os.path.join(work, "progress.jsonl")
    if _os.path.exists(path):
        with open(path, "rb") as fh:
            fh.seek(offset)
            data = fh.read(1 << 20)
        end = data.rfind(b"\n") + 1
        out["offset"] = offset + end
        for line in data[:end].splitlines():
            try:
                out["lines"].append(_json.loads(line))
            except ValueError:
                pass
    if not out["alive"]:
        try:
            with open(_os.path.join(work, "run.log"), "rb") as fh:
                fh.seek(0, 2)
                fh.seek(max(fh.tell() - 3000, 0))
                out["log"] = fh.read().decode(errors="replace")
        except OSError:
            pass
    print("@@fh-poll " + _json.dumps(out))


_fh_poll()

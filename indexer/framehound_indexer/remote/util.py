# Runs in the Colab kernel (``colab exec -f``): small file chores around uploads
# and downloads, which the CLI does in one base64 request per file, so big files
# move in parts. Env: FH_OP = mkdir | join | split | unpack, plus per-op paths.
# Prints "@@fh-util <json>".
import json as _json
import os as _os


def _fh_util():
    op = _os.environ["FH_OP"]
    out = {"ok": True}
    if op == "mkdir":
        for d in _os.environ["FH_DIRS"].split(":"):
            _os.makedirs(d, exist_ok=True)
    elif op == "join":
        # FH_DEST + FH_PARTS (count): DEST.part0..N-1 → DEST
        dest, n = _os.environ["FH_DEST"], int(_os.environ["FH_PARTS"])
        with open(dest + ".tmp", "wb") as fh:
            for i in range(n):
                part = f"{dest}.part{i}"
                with open(part, "rb") as p:
                    while chunk := p.read(1 << 22):
                        fh.write(chunk)
                _os.remove(part)
        _os.replace(dest + ".tmp", dest)
        out["bytes"] = _os.path.getsize(dest)
    elif op == "split":
        src, size = _os.environ["FH_SRC"], int(_os.environ["FH_CHUNK"])
        n = 0
        with open(src, "rb") as fh:
            while data := fh.read(size):
                with open(f"{src}.part{n}", "wb") as p:
                    p.write(data)
                n += 1
        out["parts"] = n
        out["bytes"] = _os.path.getsize(src)
    elif op == "unpack":
        import tarfile

        tar, dest = _os.environ["FH_TAR"], _os.environ["FH_DEST"]
        if not _os.path.isdir(dest):
            with tarfile.open(tar) as t:
                t.extractall(dest + ".tmp", filter="data")
            _os.replace(dest + ".tmp", dest)
    elif op == "kill":
        import signal

        try:
            with open(_os.environ["FH_PIDFILE"]) as fh:
                _os.killpg(int(fh.read().strip()), signal.SIGTERM)
        except (OSError, ValueError) as e:
            out["error"] = str(e)
    print("@@fh-util " + _json.dumps(out))


_fh_util()

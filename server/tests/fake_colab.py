"""A fake ``colab`` CLI for the ColabIndexBackend tests (design.md §11).

Each session is a directory under ``$FAKE_COLAB_DIR/vms/{name}``; any path or
``--env`` value under ``/content`` is mapped into it, so ``exec -f`` really runs
the launch/poll/util scripts, and they really start the (fake) indexer, on
"the VM". Output mimics colab-cli 0.7.4's messages. Every argv is appended to
``calls.jsonl``.

Scenario knobs (env): ``FAKE_COLAB_REJECT`` (comma-separated GPUs that ``new``
refuses with the 400 message), ``FAKE_COLAB_VANISH`` (a session name that
disappears at its first poll), ``FAKE_COLAB_BALANCE`` (starting balance),
``FAKE_COLAB_AUTH_BROKEN`` (every call fails with a 401).
"""

import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ["FAKE_COLAB_DIR"])
STATE = ROOT / "state.json"


def load() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"sessions": {}, "balance": float(os.environ.get("FAKE_COLAB_BALANCE", "100")),
            "polls": {}}


def save(st: dict) -> None:
    STATE.write_text(json.dumps(st))


def vm(name: str) -> Path:
    return ROOT / "vms" / name


def mapped(name: str, value: str) -> str:
    return str(vm(name)) + value if value.startswith("/content") else value


def line(name: str, s: dict) -> str:
    return f"[{name}] {s['endpoint']} | Hardware: {s['gpu']} | Shape: Standard | Variant: GPU"


def main(argv: list[str]) -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "calls.jsonl", "a") as fh:
        fh.write(json.dumps(argv) + "\n")
    if argv[:1] == ["--auth"]:
        argv = argv[2:]
    if os.environ.get("FAKE_COLAB_AUTH_BROKEN"):
        print("[colab] Warning: failed to fetch compute-unit info: 401 Client Error: "
              "Unauthorized")
        return 1
    cmd, rest = argv[0], argv[1:]
    opts: dict[str, list[str]] = {}
    pos: list[str] = []
    i = 0
    while i < len(rest):
        if rest[i].startswith("-"):
            opts.setdefault(rest[i], []).append(rest[i + 1])
            i += 2
        else:
            pos.append(rest[i])
            i += 1
    st = load()
    name = (opts.get("-s") or [None])[0]
    sess = st["sessions"].get(name) if name else None

    if cmd == "usage":
        print(f"Current balance: {st['balance']:.2f} compute units\n"
              f"Usage rate: {1.5 * len(st['sessions']):.2f}/hr\n"
              f"Active assignments: {len(st['sessions'])}")
        return 0
    if cmd == "sessions":
        if not st["sessions"]:
            print("[colab] No active sessions found on server.")
        for n, s in st["sessions"].items():
            print(line(n, s))
        return 0
    if cmd == "new":
        gpu = opts["--gpu"][0]
        print(f"[colab] Creating session '{name}'...")
        if gpu in os.environ.get("FAKE_COLAB_REJECT", "").split(","):
            print(f"[colab] Backend rejected accelerator '{gpu}'. You may not have quota or "
                  "entitlement for this accelerator on your account.")
            return 1
        st["sessions"][name] = {"gpu": gpu, "endpoint": f"gpu-{gpu.lower()}-s-{name}"}
        vm(name).mkdir(parents=True, exist_ok=True)
        save(st)
        print("[colab] Session READY.")
        return 0
    if cmd == "status":
        print(line(name, sess) + " | Status: IDLE" if sess else
              f"[colab] Session '{name}' not found.")
        return 0
    if cmd == "stop":
        if not sess:
            print(f"[colab] Session '{name}' not found.")
            return 0
        print(f"[colab] Stopping session '{name}'...")
        for pidfile in vm(name).rglob("pid"):
            try:
                os.killpg(int(pidfile.read_text()), signal.SIGKILL)
            except (OSError, ValueError):
                pass
        del st["sessions"][name]
        st["balance"] -= 0.4  # what the session cost
        save(st)
        print("[colab] Session terminated.")
        return 0
    if cmd == "log":
        Path(opts["-o"][0]).write_text(json.dumps({"session": name}) + "\n")
        return 0
    if sess is None:
        print(f"[colab] Session '{name}' not found.")
        return 1
    if cmd == "install":
        print(f"[colab] Installing packages on {name} (preferring uv)...")
        print("Installation Complete (via uv)!")
        return 0
    if cmd == "upload":
        local, remote = pos
        dest = Path(mapped(name, remote))
        if not dest.parent.is_dir():
            print(f"[colab] Upload failed: 404 Client Error for {remote}")
            return 1
        data = Path(local).read_bytes()
        if dest.name == "job.json":  # the pipeline reads its paths from here
            data = data.replace(b'"/content', f'"{vm(name)}/content'.encode())
        dest.write_bytes(data)
        print(f"[colab] Uploaded '{local}' to '{remote}'")
        return 0
    if cmd == "download":
        remote, local = pos
        shutil.copyfile(mapped(name, remote), local)
        print(f"[colab] Downloaded '{remote}' to '{local}'")
        return 0
    if cmd == "exec":
        script = opts["-f"][0]
        if Path(script).name == "poll.py" and os.environ.get("FAKE_COLAB_VANISH") == name:
            del st["sessions"][name]
            save(st)
            print(f"[colab] Session '{name}' appears to be lost (404/401). Cleaning up.")
            return 1
        env = dict(os.environ)
        for kv in opts.get("--env", []):
            k, _, v = kv.partition("=")
            env[k] = ":".join(mapped(name, part) for part in v.split(":"))
        p = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True,
                           check=False)
        sys.stdout.write(p.stdout + p.stderr)
        return 0  # like the real CLI: exceptions in the code don't change the exit code
    print(f"fake colab: unknown command {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

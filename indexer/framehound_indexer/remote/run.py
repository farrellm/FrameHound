"""The detached indexer process on the VM: make sure onnxruntime and ctranslate2
can use the GPU, then hand over to ``python -m framehound_indexer job.json``.

faster-whisper depends on the CPU build of onnxruntime (for its VAD), and both
builds install the same ``onnxruntime`` module, so the GPU build is swapped in
here, after ``colab install``, when a GPU is present and CUDA isn't available.

The Colab image's system CUDA can be newer (13) than the CUDA 12 that
ctranslate2 and onnxruntime-gpu link against; the image also carries CUDA 12
cuBLAS/cuDNN as pip wheels (``nvidia/*/lib``), which only need to be on the
library path, and that has to be set before the process starts.
"""

import glob
import os
import shutil
import site
import subprocess
import sys

ORT_GPU = "onnxruntime-gpu>=1.20,<2"
CHECK = "import onnxruntime as o; print('CUDAExecutionProvider' in o.get_available_providers())"


def _pip(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "-q", *args], check=False)


def main() -> None:
    job = sys.argv[1]
    if shutil.which("nvidia-smi"):
        ok = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True,
                            check=False)
        if ok.stdout.strip() != "True":
            print("installing onnxruntime-gpu", flush=True)
            _pip("uninstall", "-y", "onnxruntime", "onnxruntime-gpu")
            _pip("install", "--no-deps", "--force-reinstall", ORT_GPU)
    env = dict(os.environ)
    libs = nvidia_lib_dirs()
    if libs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join([*libs, env.get("LD_LIBRARY_PATH", "")])
    os.execve(sys.executable, [sys.executable, "-m", "framehound_indexer", job], env)


def nvidia_lib_dirs() -> list[str]:
    dirs = []
    for sp in site.getsitepackages():
        dirs += sorted(d for d in glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
                       if os.path.isdir(d))
    # The versioned bundles (nvidia/cu13) last: the per-library CUDA 12 dirs first.
    return sorted(dirs, key=lambda d: os.path.basename(os.path.dirname(d)).startswith("cu"))


if __name__ == "__main__":
    main()

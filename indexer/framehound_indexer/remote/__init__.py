"""Scripts the host runs on a Colab VM with ``colab exec -f`` (launch, poll), the
detached runner they start (run), and the VM's extra requirements. They must
not import ``framehound_indexer`` from the kernel: launch and poll run inside
the notebook kernel, where only the standard library is assumed."""

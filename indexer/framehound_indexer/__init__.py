"""FrameHound indexing pipeline and query encoders.

The api imports the encoders from here; the pipeline runs as
``python -m framehound_indexer job.json`` on a Colab VM or locally.
"""

from .lock import ModelsLock, load_lock

__all__ = ["ModelsLock", "load_lock"]
__version__ = "0.1.0"

import os

TOKEN_LENGTH = 30
# CLIP checkpoint cache directory. Machine-specific by nature (this is a local disk path,
# not something meaningful to commit as a constant for every clone/environment -- e.g. it
# does not exist at all on Colab, see notebooks/colab_entry.ipynb), so it is read from an
# environment variable with the original development machine's path kept only as a
# same-behavior-if-unset default. Set CLIP_CACHE_ROOT to override.
DOWNLOAD_ROOT = os.environ.get("CLIP_CACHE_ROOT", "D:/second_degree/deep/.cache/clip/")

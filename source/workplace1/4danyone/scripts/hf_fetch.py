"""Download one manifest entry through the official HF Xet transport."""
import json
import os
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from huggingface_hub import hf_hub_download

item = json.loads(sys.argv[1])
repository, tail = urlsplit(item["url"]).path.lstrip("/").split("/resolve/", 1)
revision, filename = tail.split("/", 1)
filename = unquote(filename)
destination = Path(item["path"])
local_dir = destination.parents[len(PurePosixPath(filename).parts) - 1]
result = hf_hub_download(repository, filename, revision=revision, local_dir=local_dir)
if Path(result).resolve() != destination.resolve():
    raise RuntimeError("Official downloader returned an unexpected destination")

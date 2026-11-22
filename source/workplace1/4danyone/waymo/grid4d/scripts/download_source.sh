#!/usr/bin/env bash
set -euo pipefail
cd /mnt/ojc/workplace1/4danyone/waymo/grid4d
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
git() { command git -c http.proxy= -c https.proxy= "$@"; }
trap 'code=$?; python -c "import json,pathlib,datetime; pathlib.Path(\"logs/source_exit.json\").write_text(json.dumps(dict(exit_code=$code, time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+\"\n\")"' EXIT
if test -d source; then
    test ! -e tmp/source_full_clone_interrupted
    mv source tmp/source_full_clone_interrupted
fi
git clone --filter=blob:none --depth=1 --no-checkout https://github.com/JiaweiXu8/Grid4D.git source
git -C source sparse-checkout set --no-cone '/*' '!*.mp4' '!*.gif' '!*.jpg' '!*.jpeg' '!*.png' '!*.pdf'
git -C source checkout --detach 3ab7b74482910d22fba7d51ac60c8cae1fe24bf1
git -C source submodule update --init --recursive --depth=1
test "$(git -C source rev-parse HEAD)" = 3ab7b74482910d22fba7d51ac60c8cae1fe24bf1
git -C source submodule status --recursive > research/source_submodules.txt
touch logs/source.complete

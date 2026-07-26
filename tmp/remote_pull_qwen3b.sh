#!/bin/sh
set -eu
cd /home/trishita/svgpatchlab-semantic
curl -fsS http://127.0.0.1:11434/api/pull \
  -H 'Content-Type: application/json' \
  -d '{"name":"qwen2.5:3b"}' \
  > qwen2.5-3b-pull.log

#!/usr/bin/env bash
# Rebuild the trace-graph visualizer's data from the mantis working tree (CPU only; no GPU).
#   docs/trace_viz/build.sh                 every robot
#   docs/trace_viz/build.sh baxter panda    some robots (the others' files are kept)
# Writes docs/trace_viz/{npz/, mantis_trace_viz.npz, web/} and mantis_trace_viz.zip at the repo root.
set -euo pipefail
cd "$(dirname "$0")/../.."
D=docs/trace_viz
export PYTHONPATH=python:tests
python "$D/export_all.py" "$D/npz" "$@"            # trace, replay log, schedules, layouts -> npz/<robot>.npz
python "$D/bundle.py" "$D/npz" "$D/mantis_trace_viz.npz"   # npz/index.json + the combined .npz
tmp=$(mktemp -d)
python "$D/npz2json.py" "$D/npz" "$tmp"            # the page's data as JSON (what the online artifact serves)
cp "$D/npz/index.json" "$tmp/index.json"
rm -rf "$D/web" && python "$D/make_offline.py" "$tmp" "$D/web"   # ... wrapped as scripts for file://
rm -rf "$tmp"
"$D/zip.sh"                                          # mantis_trace_viz.zip at the repo root
echo "built $D"

#!/usr/bin/env bash
# Package docs/trace_viz as mantis_trace_viz.zip at the repo root, to send to people without the repo:
# the visualizer, its data and the scripts, without the per-robot npz/ (the combined .npz is included).
set -euo pipefail
cd "$(dirname "$0")/../.."
rm -f mantis_trace_viz.zip
(cd docs && zip -qr ../mantis_trace_viz.zip trace_viz -x 'trace_viz/npz/*' -x '*/__pycache__/*')
echo "mantis_trace_viz.zip: $(du -h mantis_trace_viz.zip | cut -f1)"

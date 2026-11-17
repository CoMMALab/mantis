"""After export_all.py: write npz/index.json for the page, and one combined .npz for collaborators.

  python bundle.py <npz_dir> <combined.npz>

The combined file holds every robot's arrays under "<robot>/<algorithm>.<field>" and each robot's
metadata under "<robot>/meta" (UTF-8 JSON as uint8):

  z = np.load("mantis_trace_viz.npz")
  meta = json.loads(z["g1/meta"].tobytes())
  events = z["g1/traj_collision.events"]
"""
import json, os, sys
import numpy as np

src, combined = sys.argv[1], sys.argv[2]
order = ["baxter", "panda", "fr3", "ur5", "fetch", "pr2", "g1"]
index, merged = {"robots": {}}, {}
for name in order:
    path = f"{src}/{name}.npz"
    if not os.path.exists(path):
        continue
    z = np.load(path)
    meta = json.loads(z["meta"].tobytes())
    index["robots"][name] = {"n_q": meta["n_q"], "urdf": meta["urdf"], "algorithms": meta["algorithms"],
                             "skipped": meta["skipped"], "bytes": os.path.getsize(path)}
    for k in z.files:
        merged[f"{name}/{k}"] = z[k]
json.dump(index, open(f"{src}/index.json", "w"), indent=1)
np.savez_compressed(combined, **merged)
print(f"index: {len(index['robots'])} robots; {combined}: {os.path.getsize(combined) / 1e6:.1f} MB, {len(merged)} arrays")

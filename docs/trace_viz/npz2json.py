"""The page's copy of each robot .npz (the host serves no archives): web/<robot>/meta.json, and
web/<robot>/<algorithm>.json holding that algorithm's arrays under the same names, as flat lists."""
import json, os, sys
import numpy as np
src, dst = sys.argv[1], sys.argv[2]
biggest = 0
for f in sorted(os.listdir(src)):
    if not f.endswith(".npz"): continue
    robot, z = f[:-4], np.load(f"{src}/{f}")
    os.makedirs(f"{dst}/{robot}", exist_ok=True)
    meta = json.loads(z["meta"].tobytes())
    json.dump(meta, open(f"{dst}/{robot}/meta.json", "w"), separators=(",", ":"))
    for a in meta["algorithms"]:
        path = f"{dst}/{robot}/{a}.json"
        json.dump({k: z[k].ravel().tolist() for k in z.files if k.startswith(a + ".")}, open(path, "w"), separators=(",", ":"))
        biggest = max(biggest, os.path.getsize(path))
print("largest file", biggest / 1e6, "MB")

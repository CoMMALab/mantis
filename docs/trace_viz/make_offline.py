"""Wrap the page's JSON data (npz2json.py output) as scripts, so index.html works opened from disk:
each web/<path>.js registers its data in window.__MANTIS["web/<path>.json"].

  python make_offline.py <json_dir> <web_dir>
"""
import os, sys
src, dst = sys.argv[1], sys.argv[2]
for d, _, fs in os.walk(src):
    for n in fs:
        if not n.endswith(".json"): continue
        rel = os.path.relpath(os.path.join(d, n), src)
        os.makedirs(os.path.dirname(os.path.join(dst, rel)) or dst, exist_ok=True)
        body = open(os.path.join(d, n)).read()
        open(os.path.join(dst, rel[:-5] + ".js"), "w").write(f'(window.__MANTIS ||= {{}})["web/{rel}"] = {body};\n')

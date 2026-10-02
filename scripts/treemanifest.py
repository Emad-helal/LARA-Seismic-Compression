"""Content manifest of a directory tree: relative path -> (size, sha256).

Usage:  python treemanifest.py <root> <out.json> [--exclude <name>]
The manifest is written OUTSIDE the tree it describes, so hashing never adds a
file to the directory being protected.
"""
import hashlib
import json
import os
import sys

root = os.path.abspath(sys.argv[1])
out = os.path.abspath(sys.argv[2])
exclude = sys.argv[sys.argv.index("--exclude") + 1] if "--exclude" in sys.argv else None

entries = {}
for dirpath, dirnames, filenames in os.walk(root):
    if exclude:
        dirnames[:] = [d for d in dirnames if d != exclude]
    for fn in filenames:
        full = os.path.join(dirpath, fn)
        rel = os.path.relpath(full, root).replace("\\", "/")
        try:
            with open(full, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
            entries[rel] = [os.path.getsize(full), digest]
        except OSError as e:
            entries[rel] = [-1, f"unreadable: {e}"]

with open(out, "w", encoding="utf-8") as fh:
    json.dump(entries, fh, indent=0, sort_keys=True)

total = sum(v[0] for v in entries.values() if v[0] > 0)
print(f"{len(entries):>5} files  {total / (1024 ** 2):>9.1f} MB  ->  {out}")

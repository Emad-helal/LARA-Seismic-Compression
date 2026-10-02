"""Execute selected code cells of the built notebook in one namespace.

The `if __name__ == "__main__"` guard is required on Windows: DataLoader
(num_workers=2) spawns workers that re-import __main__, and without the guard
they would re-execute the whole notebook.
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT = (r"D:\Post_Doctor\Dr. Mostafa\Enhanced_Residual_Autoencoder\Dr. Omar"
       r"\Review_27092026\Final_Ablation_27092026.ipynb")


def main():
    nb = json.load(open(OUT, encoding="utf-8"))
    code = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"])
            if c["cell_type"] == "code"}
    want = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else sorted(code)
    ns = {"__name__": "__main__", "__file__": OUT}
    for i in want:
        print(f"\n{'#' * 70}\n# CELL {i}\n{'#' * 70}", flush=True)
        exec(compile(code[i], f"<cell {i}>", "exec"), ns)
    print(f"\n{'=' * 70}\ncells executed: {want}\n{'=' * 70}")


if __name__ == "__main__":
    main()

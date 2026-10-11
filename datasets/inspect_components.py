"""Print the real parquet schema + row counts of every downloaded component for a context.
    python inspect_components.py --root ../data/raw [--context NAME]
"""
import argparse
import glob
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from waymo_adapter import COMPONENTS, list_contexts, component_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"))
    ap.add_argument("--context")
    a = ap.parse_args()
    ctxs = [a.context] if a.context else list_contexts(a.root)
    if not ctxs:
        sys.exit(f"no contexts under {a.root}")
    for ctx in ctxs:
        print("=" * 100, "\ncontext", ctx)
        for comp in COMPONENTS:
            p = component_path(a.root, comp, ctx)
            if p is None:
                print(f"\n[{comp}]  ABSENT")
                continue
            f = pq.ParquetFile(p)
            print(f"\n[{comp}]  rows={f.metadata.num_rows}  file={os.path.getsize(p)/1e6:.1f} MB")
            for fld in f.schema_arrow:
                print(f"    {fld.name:70s} {fld.type}")


if __name__ == "__main__":
    main()

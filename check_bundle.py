"""check_bundle.py — does the app, as it will serve it, reproduce the paper?

`prep_data.py` does no arithmetic on purpose, so nothing between the research
repo's gates and this repo re-derives a number. This closes the last gap: it
imports `app.py` itself, values every 2025 firm through the SAME code path the
site serves — `value_target(..., "5fold")`, the fold-out curve set, the Rule D
peer set — and compares against `that_ff_pooled`, the paper's stored Rule D
prediction for that firm.

Needs the research data (`$MULTIPLES_DATA`) and so is a dev tool, like
prep_data.py; the deployed container never runs it. Run it after every
regeneration of `data/app_data.json`, before deploying.

Usage:  python check_bundle.py
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import app as A

STORE = (Path(os.environ["MULTIPLES_DATA"]) / "analyst" / "pooled_rebuild"
         / "data")
TOL = 1e-8


def main():
    worst, checked, missing = 0.0, 0, []
    for samp in ("nonmicro", "micro"):
        st = pd.read_parquet(STORE / f"logmult_{samp}_2025.parquet")
        st["gvkey"] = st["gvkey"].astype(str)
        for gv, that, fold in zip(st["gvkey"], st["that_ff_pooled"], st["fold"]):
            f = A.FIRMS.get(gv)
            if f is None:
                missing.append(gv)
                continue
            assert f["fold"] == int(fold), f"{gv}: fold {f['fold']} vs {fold}"
            v = A.value_target(f, "5fold")
            if v["fair_multiple"] is None or not np.isfinite(that):
                assert v["fair_multiple"] is None and not np.isfinite(that), gv
                continue
            d = abs(math.log(v["fair_multiple"]) - float(that))
            worst = max(worst, d)
            checked += 1
    print(f"checked {checked:,} firms; max |log fair multiple - stored "
          f"that_ff_pooled| = {worst:.2e}")
    if missing:
        print(f"NOT IN BUNDLE: {len(missing)} gvkeys, e.g. {missing[:5]}")
    ok = worst < TOL and not missing
    print("PASS — the site serves the paper's Rule D numbers" if ok
          else "FAIL — the bundle and the paper disagree")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

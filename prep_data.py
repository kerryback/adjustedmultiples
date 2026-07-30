"""prep_data.py — assemble the app's data bundle (data/app_data.json).

DEV TOOL, run locally against the multiples research workspace; the committed
data/app_data.json is what the deployed app serves (no research data or pandas
needed at runtime).

Bundles, for the 2025 cross-section:
  - firm identity + peer sets + GICS sub-industry  (artifact_data_2025.json)
  - corrected-panel characteristics, EBITDA, EV, market cap (extend2000 panels)
  - the rung-1 additive g_k appraisal curves, SEPARATELY for non-micro and micro

Microcaps and non-microcaps get their OWN curve set and (within-sample) peers.
If the micro curves haven't been generated yet, the non-micro curves stand in as a
labelled placeholder (micro_curves_placeholder=true) until the micro run produces them.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path("/Users/kerryback/repos/multiples/workspace")
OUT = Path(__file__).parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

NONMICRO_CURVES = WORK / "logmult" / "results" / "gk_curves_logmult_2001_2025.parquet"
MICRO_CURVES = WORK / "logmult_micro" / "results" / "gk_curves_micro_2001_2025.parquet"

FEATURES = [
    ("f_ebitdamargin", "EBITDA margin"), ("f_grossmargin", "Gross margin"),
    ("f_roa", "Return on assets (EBITDA/assets)"), ("f_assetturn", "Asset turnover"),
    ("f_salegrow", "Sales growth, 1yr (log)"), ("f_salegrow3", "Sales growth, 3yr"),
    ("f_assetgrow3", "Asset growth, 3yr"), ("f_ltg", "Long-term growth forecast"),
    ("f_epsgrow", "Forward EPS growth"), ("has_ibes", "IBES coverage"),
    ("f_rdsale", "R&D / sales"), ("f_capxsale", "Capex / sales"),
    ("f_leverage", "Leverage"), ("f_cashassets", "Cash / assets"),
    ("f_tangibility", "Tangibility"), ("f_accruals", "Accruals"),
    ("f_profitflag", "Profit-sign flag"), ("f_ebitdapos", "EBITDA-positive flag"),
    ("f_age", "Firm age (years)"), ("log_at", "Log assets"), ("log_sale", "Log sales"),
]
FEAT_KEYS = [k for k, _ in FEATURES]


def load_curves(path: Path) -> dict:
    gk = pd.read_parquet(path)
    c = {}
    for feat, sub in gk.groupby("feature"):
        sub = sub.sort_values("x_grid")
        c[feat] = {"x": sub["x_grid"].astype(float).tolist(),
                   "g": sub["g_k"].astype(float).tolist()}
    return c


def main():
    art = json.loads((WORK / "logmult" / "artifact_data_2025.json").read_text())
    idmap = {f["gvkey"]: f for f in art["firms"]}

    frames = []
    for samp, path in [("nonmicro", "extend2000/data/panel_nonmicro.parquet"),
                       ("micro", "extend2000/data/panel_micro.parquet")]:
        p = pd.read_parquet(WORK / path)
        p = p[(p["valyear"] == 2025) & (p["ebitda_yield"].astype(float) > 0)].copy()
        p["gvkey"] = p["gvkey"].astype(str)
        p["sample"] = samp
        frames.append(p)
    panel = pd.concat(frames, ignore_index=True)

    nonmicro = load_curves(NONMICRO_CURVES)
    if MICRO_CURVES.exists():
        micro = load_curves(MICRO_CURVES)
        placeholder = False
    else:
        micro = nonmicro  # placeholder until the micro curves are generated
        placeholder = True
    curves = {"nonmicro": nonmicro, "micro": micro}

    firms = {}
    for _, r in panel.iterrows():
        gv = r["gvkey"]; meta = idmap.get(gv)
        if meta is None or not meta.get("ticker"):
            continue
        ebitda, ev = float(r["ebitda"]), float(r["ev"])
        if not (np.isfinite(ebitda) and np.isfinite(ev) and ebitda > 0 and ev > 0):
            continue
        chars = {}
        for k in FEAT_KEYS:
            v = r.get(k)
            chars[k] = None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)
        firms[gv] = {
            "gvkey": gv, "ticker": meta["ticker"], "name": meta["name"],
            "sample": r["sample"], "subindustry": meta.get("subindustry", ""),
            "gsubind": meta.get("gsubind", ""),
            "mktcap_b": round(float(r["mktcap"]) / 1000.0, 3),
            "ebitda": round(ebitda, 3), "ev": round(ev, 3),
            "multiple": round(ev / ebitda, 4),
            "chars": chars,
            "peers": [g for g in meta.get("peers", []) if g in idmap],
        }
    have = set(firms)
    for f in firms.values():
        f["peers"] = [g for g in f["peers"] if g in have]

    bundle = {
        "year": 2025,
        "micro_curves_placeholder": placeholder,
        "features": [{"key": k, "label": lbl} for k, lbl in FEATURES],
        "curves": curves,
        "firms": list(firms.values()),
    }
    (OUT / "app_data.json").write_text(json.dumps(bundle))
    nsamp = {s: sum(1 for f in firms.values() if f["sample"] == s) for s in ("nonmicro", "micro")}
    print(f"[prep] wrote {OUT/'app_data.json'}: {len(firms)} firms {nsamp}, "
          f"micro_curves_placeholder={placeholder}")


if __name__ == "__main__":
    main()

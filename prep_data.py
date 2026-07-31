"""prep_data.py — assemble the valuation app's data bundle.

Combines, for the 2025 cross-section:
  - firm identity + peer sets + GICS sub-industry  (from artifact_data_2025.json)
  - CORRECTED-panel characteristics, EBITDA, EV, market cap (from extend2000 panels)
  - the rung-1 additive g_k appraisal curves (from gk_curves parquet)
into workspace/valuation_app/data/app_data.json, which app.py loads at startup.

The app values a target firm by the adjusted-multiple estimator (rung 1, lambda=1):
  adjusted multiple of peer j (on target i's terms) = m_j * exp( sum_k [g_k(x_i) - g_k(x_j)] )
  fair multiple of i = median over peers of the adjusted peer multiples
  EV_fair = fair multiple * EBITDA_i
Industry is handled by matching (peers are same-industry), so the 21 additive curves carry
the adjustment. Re-run this after the pipeline refreshes the panel / curves.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/kerryback/repos/multiples/workspace")
OUT = ROOT / "valuation_app" / "data"
OUT.mkdir(parents=True, exist_ok=True)

# 21 additive characteristics in the model g, with display labels (order = presentation order)
FEATURES = [
    ("f_ebitdamargin", "EBITDA margin"),
    ("f_grossmargin", "Gross margin"),
    ("f_roa", "Return on assets (EBITDA/assets)"),
    ("f_assetturn", "Asset turnover"),
    ("f_salegrow", "Sales growth, 1yr (log)"),
    ("f_salegrow3", "Sales growth, 3yr"),
    ("f_assetgrow3", "Asset growth, 3yr"),
    ("f_ltg", "Long-term growth forecast"),
    ("f_epsgrow", "Forward EPS growth"),
    ("has_ibes", "IBES coverage"),
    ("f_rdsale", "R&D / sales"),
    ("f_capxsale", "Capex / sales"),
    ("f_leverage", "Leverage"),
    ("f_cashassets", "Cash / assets"),
    ("f_tangibility", "Tangibility"),
    ("f_accruals", "Accruals"),
    ("f_profitflag", "Profit-sign flag"),
    ("f_ebitdapos", "EBITDA-positive flag"),
    ("f_age", "Firm age (years)"),
    ("log_at", "Log assets"),
    ("log_sale", "Log sales"),
]
FEAT_KEYS = [k for k, _ in FEATURES]


def main():
    # --- identity + peers (names, tickers, sub-industry, peer gvkeys) ---
    art = json.loads((ROOT / "logmult" / "artifact_data_2025.json").read_text())
    idmap = {f["gvkey"]: f for f in art["firms"]}

    # --- corrected-panel characteristics for the 2025 positive-EBITDA cross-section ---
    frames = []
    for samp, path in [("nonmicro", "extend2000/data/panel_nonmicro.parquet"),
                       ("micro", "extend2000/data/panel_micro.parquet")]:
        p = pd.read_parquet(ROOT / path)
        p = p[(p["valyear"] == 2025) & (p["ebitda_yield"].astype(float) > 0)].copy()
        p["gvkey"] = p["gvkey"].astype(str)
        p["sample"] = samp
        frames.append(p)
    panel = pd.concat(frames, ignore_index=True)

    # --- g_k curves for 2025 from the canonical store (single source of truth): the
    #     full-sample (1-fold) fit + the 5 cross-fit folds, plus the saved fold membership ---
    STORE = ROOT / "logmult" / "results" / "gk_store"

    def curves_dict(df):
        out = {}
        for feat, sub in df.groupby("feature"):
            sub = sub.sort_values("x_grid")
            out[feat] = {"x": sub["x_grid"].astype(float).tolist(),
                         "g": sub["g_k"].astype(float).tolist()}
        return out

    gk = pd.read_parquet(STORE / "gk_functions.parquet")
    gk = gk[gk["year"] == 2025]
    curves_full = curves_dict(gk[gk["fold"] == "full"])
    curves_folds = {int(f): curves_dict(g) for f, g in gk[gk["fold"] != "full"].groupby("fold")}
    fm = pd.read_parquet(STORE / "fold_membership.parquet")
    fm = fm[fm["year"] == 2025]
    fold_assign = dict(zip(fm["gvkey"].astype(str), fm["fold"].astype(int)))

    # --- assemble firm records (only firms present in BOTH panel and identity map) ---
    firms = {}
    for _, r in panel.iterrows():
        gv = r["gvkey"]
        meta = idmap.get(gv)
        if meta is None or meta.get("ticker") in (None, ""):
            continue
        ebitda = float(r["ebitda"]); ev = float(r["ev"])
        if not (np.isfinite(ebitda) and np.isfinite(ev) and ebitda > 0 and ev > 0):
            continue
        chars = {}
        for k in FEAT_KEYS:
            v = r.get(k)
            chars[k] = None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)
        firms[gv] = {
            "gvkey": gv,
            "ticker": meta["ticker"],
            "name": meta["name"],
            "sample": r["sample"],
            "subindustry": meta.get("subindustry", ""),
            "gsubind": meta.get("gsubind", ""),
            "mktcap_b": round(float(r["mktcap"]) / 1000.0, 3),
            "ebitda": round(ebitda / 1000.0, 4),   # $bn (panel stores $mm), consistent with mktcap_b
            "ev": round(ev / 1000.0, 4),            # $bn
            "multiple": round(ev / ebitda, 4),      # ratio, unit-independent
            "fold": int(fold_assign.get(gv, -1)),   # cross-fit fold (non-micro 0-4; -1 = no fold)
            "chars": chars,
            "peers": [g for g in meta.get("peers", []) if g in idmap],
        }
    # keep only peers that we actually have full records (chars) for
    have = set(firms)
    for f in firms.values():
        f["peers"] = [g for g in f["peers"] if g in have]

    bundle = {
        "year": 2025,
        "features": [{"key": k, "label": lbl} for k, lbl in FEATURES],
        "curves_full": curves_full,
        "curves_folds": {str(f): c for f, c in curves_folds.items()},
        "n_folds": len(curves_folds),
        "firms": list(firms.values()),
    }
    (OUT / "app_data.json").write_text(json.dumps(bundle))
    n_peers = np.median([len(f["peers"]) for f in firms.values()])
    nf = sum(1 for f in firms.values() if f["fold"] >= 0)
    print(f"[prep] wrote {OUT/'app_data.json'}: {len(firms)} firms ({nf} with a fold), "
          f"median peers={n_peers:.0f}, full + {len(curves_folds)} fold curve sets")
    # examples
    for tk in ("AAPL", "CALM", "DPZ"):
        m = next((f for f in firms.values() if f["ticker"] == tk), None)
        if m:
            print(f"  {tk}: {m['name']} mult={m['multiple']} peers={len(m['peers'])}")


if __name__ == "__main__":
    main()

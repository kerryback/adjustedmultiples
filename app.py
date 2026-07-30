"""Adjusted-multiple valuation app (FastAPI).

Values a firm the way the paper's rung-1 estimator does: take its size-and-industry
peers, adjust each peer's EV/EBITDA multiple for how the peer differs from the target
on the operating characteristics (via the additive g_k appraisal curves), and take the
median of the adjusted peer multiples as the fair multiple.

  adjusted multiple of peer j (on target i's terms) = m_j * exp( sum_k [g_k(x_i) - g_k(x_j)] )
  fair multiple of i = median_j (adjusted peer multiples)
  fair EV = fair multiple * EBITDA_i

Microcaps and non-microcaps use SEPARATE g_k curve sets (fit on their own sample) and
peer sets (matched within their own universe). The curve set is chosen by the target's
sample; its peers are in the same sample.

Data is pre-built into data/app_data.json (see prep_data.py). Run:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""
import json
import math
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).parent
DATA = json.loads((APP_DIR / "data" / "app_data.json").read_text())
FIRMS = {f["gvkey"]: f for f in DATA["firms"]}
BY_TICKER = {f["ticker"].upper(): f for f in DATA["firms"] if f.get("ticker")}
CURVES = DATA["curves"]          # {"nonmicro": {feat:{x,g}}, "micro": {feat:{x,g}}}
FEATURES = DATA["features"]      # [{key,label}, ...] in presentation order
MICRO_PLACEHOLDER = DATA.get("micro_curves_placeholder", False)

app = FastAPI(title="Adjusted-Multiple Valuation")


def gk(sample: str, feat: str, x):
    """g_k(x) for the given sample's curve set, interpolated at value x."""
    if x is None:
        return None
    cs = CURVES.get(sample) or CURVES.get("nonmicro")
    c = cs.get(feat)
    if not c or not c["x"]:
        return 0.0
    return float(np.interp(x, c["x"], c["g"]))  # np.interp clamps outside the grid


def value_target(t: dict) -> dict:
    sample = t.get("sample", "nonmicro")
    peers = [FIRMS[g] for g in t.get("peers", []) if g in FIRMS]
    peer_out, adj = [], []
    for j in peers:
        bd, tot = [], 0.0
        for ft in FEATURES:
            k = ft["key"]
            xi, xj = t["chars"].get(k), j["chars"].get(k)
            gi, gj = gk(sample, k, xi), gk(sample, k, xj)
            dg = (gi - gj) if (gi is not None and gj is not None) else 0.0
            tot += dg
            bd.append({"key": k, "label": ft["label"], "xi": xi, "xj": xj,
                       "gi": gi, "gj": gj, "dg": dg})
        adjm = j["multiple"] * math.exp(tot)
        adj.append(adjm)
        bd_sorted = sorted(bd, key=lambda r: -abs(r["dg"]))
        peer_out.append({
            "gvkey": j["gvkey"], "ticker": j["ticker"], "name": j["name"],
            "subindustry": j["subindustry"], "mktcap_b": j["mktcap_b"],
            "same_industry": j.get("gsubind") == t.get("gsubind"),
            "actual_multiple": j["multiple"], "log_adjustment": tot,
            "adjustment_factor": math.exp(tot), "adjusted_multiple": adjm,
            "breakdown": bd_sorted,
        })
    peer_out.sort(key=lambda p: -p["mktcap_b"])
    fair = float(np.median(adj)) if adj else None
    ev_fair = (fair * t["ebitda"]) if fair is not None else None
    return {"fair_multiple": fair, "ev_fair": ev_fair, "peers": peer_out}


@app.get("/api/meta")
def meta():
    return {"year": DATA.get("year"), "n_firms": len(DATA["firms"]),
            "micro_curves_placeholder": MICRO_PLACEHOLDER}


@app.get("/api/tickers")
def tickers():
    out = [{"ticker": f["ticker"], "name": f["name"], "sample": f["sample"],
            "mktcap_b": f["mktcap_b"]}
           for f in DATA["firms"] if f.get("ticker")]
    out.sort(key=lambda r: -r["mktcap_b"])
    return out


@app.get("/api/valuation/{ticker}")
def valuation(ticker: str):
    t = BY_TICKER.get(ticker.strip().upper())
    if t is None:
        raise HTTPException(status_code=404,
                            detail=f"No firm with ticker '{ticker}' in the {DATA.get('year')} sample.")
    v = value_target(t)
    ev_fair = v["ev_fair"]
    return {
        "target": {
            "gvkey": t["gvkey"], "ticker": t["ticker"], "name": t["name"],
            "sample": t["sample"], "subindustry": t["subindustry"],
            "mktcap_b": t["mktcap_b"], "ebitda": t["ebitda"], "ev": t["ev"],
            "actual_multiple": t["multiple"],
        },
        "fair_multiple": v["fair_multiple"],
        "ev_fair": ev_fair,
        "actual_ev": t["ev"],
        "actual_multiple": t["multiple"],
        "pct_vs_fair": (t["ev"] / ev_fair - 1.0) if ev_fair else None,
        "n_peers": len(v["peers"]),
        "peers": v["peers"],
    }


@app.get("/")
def index():
    return FileResponse(APP_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

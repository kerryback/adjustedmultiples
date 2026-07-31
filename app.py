"""Adjusted-multiple valuation app (FastAPI).

Values a firm the way the paper's rung-1 estimator does: take its size-and-industry
peers, adjust each peer's EV/EBITDA multiple for how the peer differs from the target
on the operating characteristics (via the additive g_k appraisal curves), and take the
median of the adjusted peer multiples as the fair multiple.

  adjusted multiple of peer j (on target i's terms) = m_j * exp( sum_k [g_k(x_i) - g_k(x_j)] )
  fair multiple of i = median_j (adjusted peer multiples)
  fair EV = fair multiple * EBITDA_i

Run:  uvicorn app:app --reload --port 8100   (from workspace/valuation_app/)
"""
import json
import math
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).parent
DATA = json.loads((APP_DIR / "data" / "app_data.json").read_text())
FIRMS = {f["gvkey"]: f for f in DATA["firms"]}
BY_TICKER = {f["ticker"].upper(): f for f in DATA["firms"] if f.get("ticker")}
CURVES = DATA["curves"]      # {sample: {"full": {feat:{x,g}}, "folds": {"0".."4": {feat:{x,g}}}}}
FEATURES = DATA["features"]  # [{key,label}, ...] in presentation order

# ML-frontier (squared-error direct-general-g GBM) peer-weight matrices, per sample, both on 2025:
#   full  = single in-sample fit (has self-weights);  5fold = cross-fit (no self-weight).
# The paper estimates each sample separately, so a ticker routes to its own sample's matrices.
_FRONTIER_FILE = APP_DIR / "data" / "ml_frontier.npz"
F_EPS = 1e-6
F_S, F_FAIR, F_TICK, F_MULT, F_IDX = {}, {}, {}, {}, {}
if _FRONTIER_FILE.exists():
    _fz = np.load(_FRONTIER_FILE, allow_pickle=False)
    for _s in (str(s) for s in _fz["samples"]):
        F_S[_s] = {"full": _fz[f"S_full_{_s}"], "5fold": _fz[f"S_cv_{_s}"]}
        F_FAIR[_s] = {"full": _fz[f"fair_full_{_s}"], "5fold": _fz[f"fair_cv_{_s}"]}
        F_TICK[_s] = [str(t) for t in _fz[f"tickers_{_s}"]]
        F_MULT[_s] = _fz[f"multiple_{_s}"]
        F_IDX[_s] = {t.upper(): i for i, t in enumerate(F_TICK[_s]) if t}

app = FastAPI(title="Adjusted-Multiple Valuation")


def gk(cs: dict, feat: str, x):
    """g_k(x) from curve set cs: interpolate the appraisal curve for `feat` at value x."""
    if x is None:
        return None
    c = cs.get(feat)
    if not c or not c["x"]:
        return 0.0
    return float(np.interp(x, c["x"], c["g"]))  # np.interp clamps outside the grid


def curves_for(t: dict, method: str) -> dict:
    """The paper applies the target's fold-out model g_{-f} to the target AND its peers,
    from the target's OWN sample (non-micro or micro). Full-sample (or a firm without a
    fold) uses that sample's single 1-fold curves."""
    cs = CURVES[t["sample"]]
    if method == "5fold" and t.get("fold", -1) >= 0:
        return cs["folds"][str(t["fold"])]
    return cs["full"]


def value_target(t: dict, method: str) -> dict:
    cs = curves_for(t, method)
    peers = [FIRMS[g] for g in t.get("peers", []) if g in FIRMS]
    peer_out, adj = [], []
    for j in peers:
        bd, tot = [], 0.0
        for ft in FEATURES:
            k = ft["key"]
            xi, xj = t["chars"].get(k), j["chars"].get(k)
            gi, gj = gk(cs, k, xi), gk(cs, k, xj)
            dg = (gi - gj) if (gi is not None and gj is not None) else 0.0
            tot += dg
            bd.append({"key": k, "label": ft["label"], "xi": xi, "xj": xj,
                       "gi": gi, "gj": gj, "dg": dg})
        adjm = j["multiple"] * math.exp(tot)
        adj.append(adjm)
        # rank the breakdown rows by absolute contribution for display
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


@app.get("/api/tickers")
def tickers():
    out = [{"ticker": f["ticker"], "name": f["name"], "sample": f["sample"],
            "mktcap_b": f["mktcap_b"]}
           for f in DATA["firms"] if f.get("ticker")]
    out.sort(key=lambda r: -r["mktcap_b"])
    return out


def _method(m: str) -> str:
    return m if m in ("full", "5fold") else "5fold"


@app.get("/api/valuation/{ticker}")
def valuation(ticker: str, method: str = "5fold"):
    method = _method(method)
    t = BY_TICKER.get(ticker.strip().upper())
    if t is None:
        raise HTTPException(status_code=404, detail=f"No firm with ticker '{ticker}' in the 2025 sample.")
    v = value_target(t, method)
    ev_fair = v["ev_fair"]
    return {
        "method": method, "fold": t.get("fold", -1),
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
        # >0 means the market prices it ABOVE its adjusted-multiple fair value (rich)
        "pct_vs_fair": (t["ev"] / ev_fair - 1.0) if ev_fair else None,
        "n_peers": len(v["peers"]),
        "peers": v["peers"],
    }


@app.get("/api/frontier/{ticker}")
def frontier(ticker: str, method: str = "5fold"):
    method = _method(method)
    key = ticker.strip().upper()
    ti = BY_TICKER.get(key)
    if ti is None:
        raise HTTPException(status_code=404, detail=f"No firm with ticker '{ticker}' in the 2025 sample.")
    s = ti["sample"]                       # route to the firm's own sample's frontier
    i = F_IDX.get(s, {}).get(key)
    if i is None:
        raise HTTPException(status_code=404,
                            detail=f"No machine-learning frontier decomposition for '{ticker}'.")
    tick, mult, fair = F_TICK[s], F_MULT[s], F_FAIR[s]
    w = F_S[s][method][i]
    peers = []
    for j in np.argsort(-np.abs(w)):
        j = int(j)
        if j == i or abs(float(w[j])) < F_EPS:
            continue
        tk = tick[j]
        meta = BY_TICKER.get(tk.upper(), {})
        peers.append({"ticker": tk, "name": meta.get("name"),
                      "subindustry": meta.get("subindustry"),
                      "multiple": round(float(mult[j]), 2),
                      "weight": round(float(w[j]), 6)})
    return {"method": method, "sample": s, "name": ti.get("name"), "subindustry": ti.get("subindustry"),
            "actual_multiple": round(float(mult[i]), 2),
            "fair_multiple": round(float(fair[method][i]), 2),
            "self_weight": round(float(w[i]), 6),
            "n_nonzero": len(peers), "peers": peers}


@app.get("/")
def index():
    return FileResponse(APP_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

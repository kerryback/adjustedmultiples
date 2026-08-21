"""Adjusted-multiple valuation app (FastAPI).

Values a firm the way the paper's rung-1 estimator does: take its peers, adjust each
peer's EV/EBITDA multiple for how the peer differs from the target on the operating
characteristics (via the additive g_k appraisal curves), and take the median of the
adjusted peer multiples as the fair multiple.

  adjusted multiple of peer j (on target i's terms) = m_j * exp( sum_k [g_k(x_i) - g_k(x_j)] )
  fair multiple of i = median_j (adjusted peer multiples), the median taken in logs
  fair EV = fair multiple * EBITDA_i

PEER RULE D (method_spec.md sec. 13d). The ladder is unchanged — GICS sector
pre-filter, four ring tiers, ten (ring, size-factor) steps over factors 2/3/4/5,
five to ten peers — but the candidate pool is the WHOLE April-30 cross-section
rather than the target's own size universe. A microcap can be a peer of a large
firm and the other way round. The curves stay per universe: a target is always
rated by its own universe's g, whichever universe each peer came from.

The bundle carries every additive piece of g — the exact piecewise cubics per
characteristic, the flag effects, the gsubind class effects, and the value the
design pipeline assigns to a MISSING characteristic — so what this app shows is
the estimator, not an approximation of it. The research repo's site builder
asserts that: it reproduces the paper's stored prediction for every firm in both
samples from this exact bundle.

Run:  uvicorn app:app --reload --port 8100
"""
import bisect
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
CURVES = DATA["curves"]      # {sample: {"full": SET, "folds": {"0".."4": SET}}}
FEATURES = DATA["features"]  # [{key,label}, ...] in presentation order
# a curve SET is {"features": {feat: piecewise cubic | flag endpoints},
#                 "g_missing": {feat: value},          <- NOT zero; see gk()
#                 "classes": {gsubind: value}, "class_unseen": value}

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
    """g_k(x) from curve set cs — the fitted appraisal curve for `feat` at value x.

    The curve is stored as the exact piecewise cubic the spline block is, one
    cubic per knot interval in the local coordinate u = (x - left)/(right - left),
    so this reproduces the estimator rather than interpolating a picture of it.
    Outside the block's training range the value is clamped to the boundary,
    which is what the design pipeline itself does — that clamping is how a
    microcap peer's log assets are rated on the large-firm curve.

    A MISSING characteristic is not a zero adjustment: the pipeline maps it to
    the mean observed basis row, and `g_missing` is that value.
    """
    c = cs["features"].get(feat)
    if c is None or c["kind"] == "none":
        return 0.0
    if x is None:
        return float(cs["g_missing"].get(feat, 0.0))
    if c["kind"] == "flag":
        return float(c["g0"] + (c["g1"] - c["g0"]) * float(x))
    brk = c["brk"]
    xc = min(max(float(x), c["lo"]), c["hi"])
    i = min(max(bisect.bisect_right(brk, xc) - 1, 0), len(brk) - 2)
    u = (xc - brk[i]) / (brk[i + 1] - brk[i])
    a, b, c2, c3 = c["coef"][i]
    return float(a + u * (b + u * (c2 + u * c3)))


def g_class(cs: dict, gsubind) -> float:
    """The GICS sub-industry class effect — a piece of g like any other. It
    cancels between a target and a peer in the same sub-industry and does NOT
    cancel otherwise, which is most of the peer set once the ladder widens."""
    return float(cs["classes"].get(str(gsubind), cs["class_unseen"]))


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
            dg = gi - gj
            tot += dg
            bd.append({"key": k, "label": ft["label"], "xi": xi, "xj": xj,
                       "gi": gi, "gj": gj, "dg": dg})
        gi, gj = g_class(cs, t["gsubind"]), g_class(cs, j["gsubind"])
        tot += gi - gj
        bd.append({"key": "gsubind", "label": "Sub-industry effect",
                   "xi": None, "xj": None, "si": t["subindustry"],
                   "sj": j["subindustry"], "gi": gi, "gj": gj, "dg": gi - gj})
        adjm = j["multiple"] * math.exp(tot)
        adj.append(math.log(adjm))
        # rank the breakdown rows by absolute contribution for display
        bd_sorted = sorted(bd, key=lambda r: -abs(r["dg"]))
        peer_out.append({
            "gvkey": j["gvkey"], "ticker": j["ticker"], "name": j["name"],
            "subindustry": j["subindustry"], "mktcap_b": j["mktcap_b"],
            "same_industry": j.get("gsubind") == t.get("gsubind"),
            # Rule D: the ladder draws from the whole cross-section, so a peer
            # can come from the other size universe
            "other_universe": j.get("sample") != t.get("sample"),
            "actual_multiple": j["multiple"], "log_adjustment": tot,
            "adjustment_factor": math.exp(tot), "adjusted_multiple": adjm,
            "breakdown": bd_sorted,
        })
    peer_out.sort(key=lambda p: -p["mktcap_b"])
    # The median is taken in LOGS, which is where the estimator takes it:
    # t_hat_i = g(x_i) + median_j [ t_j - g(x_j) ]. For an odd-sized peer set
    # the two agree; for an even-sized one the median of the levels averages
    # the two middle multiples arithmetically and the paper's averages them
    # geometrically, and the gap reaches a tenth of a log point on a two-peer
    # set. Half the peer sets are even-sized, so this is not a corner case.
    fair = float(math.exp(np.median(adj))) if adj else None
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
    return m if m in ("full", "5fold") else "full"


@app.get("/api/valuation/{ticker}")
def valuation(ticker: str, method: str = "full"):
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
def frontier(ticker: str, method: str = "full"):
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

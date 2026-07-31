"""build_ml_frontier.py — ML-frontier (direct general-g LightGBM) peer-weight view for the app.

Our machine-learning frontier is a LightGBM that predicts the log multiple t=log(EV/EBITDA)
directly from the 21 operating characteristics + GICS sub-industry (the paper's "direct
method, general g"). Following Geertsema-Lu (App A.3), a squared-error gradient-boosted
prediction can be written EXACTLY as a weighted average of the training firms' target
multiples, with weights built from how often the target and each firm co-occupy the same
LightGBM leaf across boosting iterations (scaled by the learning rate, normalized by leaf
size + reg_lambda). We expose those weights per firm.

Model = the paper's frontier architecture (same 21+1 features, 400 trees, lr 0.05,
num_leaves 31, min_child_samples 50, reg_lambda 1), with TWO changes required to make the
decomposition an EXACT, reproducible weighted average of peer multiples:
  * objective = "regression" (squared error) instead of regression_l1 — the median-loss
    frontier is not a linear smoother of the target, so its prediction cannot be written as
    a weighted average; L2 can. (L1 vs L2 fitted t correlate ~0.94 on 2025; reported below.)
  * subsample = colsample_bytree = 1 (no bagging) — so each tree's leaf value is the mean
    over ALL its leaf-mates, making the leaf-membership weights exact.
Single full-sample fit on the 2025 non-micro positive-EBITDA cross-section (a reporting fit,
like the g_k curves), so each firm's fair multiple is a weighted average of the other 2025
firms' multiples.

Outputs (valuation_app/data/):
  ml_frontier.npz         COMPACT app payload: weight matrix S (float32, n x n), tickers (firm
                          order), per-firm EV/EBITDA multiple and frontier fair multiple. The app
                          slices a ticker's row on demand and joins names from app_data.json.
  ml_frontier_leaves.npz  raw leaf memberships (n x n_trees), weight matrix S, learning_rate,
                          gvkeys — the saved leaf memberships + weights + learning rate
                          (LOCAL/reproducibility only; gitignored, not deployed).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/Users/kerryback/repos/multiples/workspace")
PANEL = ROOT / "extend2000/data/panel_nonmicro.parquet"
ARTIFACT = ROOT / "logmult/artifact_data_2025.json"   # ticker/name/subindustry per gvkey
OUT = ROOT / "valuation_app/data"

SPLINE_FEATS = ["f_grossmargin", "f_ebitdamargin", "f_roa", "f_assetturn", "f_salegrow",
                "f_salegrow3", "f_assetgrow3", "f_ltg", "f_epsgrow", "has_ibes",
                "f_rdsale", "f_capxsale", "f_leverage", "f_cashassets", "f_tangibility",
                "f_accruals", "f_profitflag", "f_ebitdapos", "f_age", "log_at", "log_sale"]
CAT_FEAT = "gsubind_code"
LGB_FEATS = SPLINE_FEATS + [CAT_FEAT]
LR = 0.05
LAMBDA = 1.0     # reg_lambda -> leaf value = lr * sum(residual)/(n_leaf + lambda)
N_TREES = 400
EPS = 1e-6       # weight magnitude below which we treat a firm as zero-weight

# ---- 2025 cross-section ----
d = pd.read_parquet(PANEL)
d = d[(d.valyear == 2025) & (d.ebitda_yield > 0)].reset_index(drop=True).copy()
d["t"] = -np.log(d["ebitda_yield"])           # log(EV/EBITDA)
d["multiple"] = 1.0 / d["ebitda_yield"]       # EV/EBITDA
d[CAT_FEAT] = d["gsubind"].astype("category").cat.codes.astype(int)
gvk = d["gvkey"].astype(str).to_numpy()
t = d["t"].to_numpy(float)
n = len(d)
print(f"2025 non-micro positive-EBITDA firms: {n}")

X = d[LGB_FEATS].copy()
X[CAT_FEAT] = X[CAT_FEAT].astype("category")

# ---- fit the frontier model (L2, no bagging) ----
model = lgb.LGBMRegressor(objective="regression", n_estimators=N_TREES, learning_rate=LR,
                          num_leaves=31, min_child_samples=50, subsample=1.0,
                          colsample_bytree=1.0, reg_lambda=LAMBDA, random_state=42,
                          n_jobs=-1, verbosity=-1, deterministic=True, force_row_wise=True)
model.fit(X, t, categorical_feature=[CAT_FEAT])
leaves = model.predict(X, pred_leaf=True)      # (n, N_TREES) leaf index per tree
raw = model.predict(X, raw_score=True)         # fitted log multiple (frontier prediction)

# sanity vs the paper's L1 frontier fitted on the same 2025 firms
mL1 = lgb.LGBMRegressor(objective="regression_l1", n_estimators=N_TREES, learning_rate=LR,
                        num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=-1,
                        verbosity=-1, deterministic=True, force_row_wise=True)
mL1.fit(X, t, categorical_feature=[CAT_FEAT])
print(f"corr(L2 fit, L1 frontier fit) on 2025 = {np.corrcoef(raw, mL1.predict(X))[0,1]:.4f}")

# ---- exact leaf-membership weight matrix S:  raw = S @ t,  rows sum ~1 ----
# S = B + sum_m lr * A_m * (prod_{l<m}(I - lr A_l)) (I - B),  A_m = leaf-mean(+lambda) operator
B = np.full((n, n), 1.0 / n)
S = B.copy()
Rop = np.eye(n) - B                            # residual operator after the mean init
for m in range(N_TREES):
    lv = leaves[:, m]
    AR = np.empty((n, n))
    for lf in np.unique(lv):
        idx = np.where(lv == lf)[0]
        AR[idx, :] = Rop[idx, :].sum(axis=0) / (len(idx) + LAMBDA)
    S += LR * AR
    Rop -= LR * AR

recon = S @ t
print(f"reconstruction max|S t - raw| = {np.max(np.abs(recon - raw)):.2e}  "
      f"(mean {np.mean(np.abs(recon - raw)):.2e})")
print(f"row-sum of S: min {S.sum(1).min():.4f} max {S.sum(1).max():.4f} (should be ~1)")

# ---- identity: ticker / name / subindustry (name/subindustry not stored; app joins them) ----
firms = {r["gvkey"]: r for r in json.load(open(ARTIFACT))["firms"]}
tick = np.array([firms.get(g, {}).get("ticker") or "" for g in gvk], dtype=object)
name = np.array([firms.get(g, {}).get("name") for g in gvk], dtype=object)
subind = np.array([firms.get(g, {}).get("subindustry") for g in gvk], dtype=object)
mult = d["multiple"].to_numpy(float)
fair = np.exp(raw)

# ---- COMPACT app payload: the weight matrix + per-firm arrays; the app computes each
# ticker's peer list on demand and joins names from app_data.json (avoids a 300MB+ JSON
# that duplicates names across 1474 x 1472 pairs). S as float32 (~8.7MB).
np.savez_compressed(OUT / "ml_frontier.npz",
                    S=S.astype(np.float32),
                    tickers=tick.astype(str),
                    multiple=mult.astype(np.float32),
                    fair=fair.astype(np.float32))
# ---- reproducibility archive (LOCAL only; gitignored, not deployed): raw leaf memberships ----
np.savez_compressed(OUT / "ml_frontier_leaves.npz", leaves=leaves, S=S, learning_rate=LR,
                    gvkeys=gvk, tickers=tick.astype(str))
print(f"\nwrote ml_frontier.npz (app payload) + ml_frontier_leaves.npz (reproducibility)")

# ---- worked example ----
for ex in ["AAPL"]:
    idx = np.where(tick == ex)[0]
    if len(idx):
        i = int(idx[0]); w = S[i, :]
        keep = [j for j in np.argsort(-np.abs(w)) if j != i and abs(w[j]) >= EPS and tick[j]]
        top = keep[:12]
        cum = (abs(w[i]) + sum(abs(w[j]) for j in top)) / (abs(w[i]) + sum(abs(w[j]) for j in keep))
        print(f"\n{ex}: {name[i]} | {subind[i]} | actual {mult[i]:.1f}x -> frontier fair {fair[i]:.1f}x")
        print(f"  self-weight {w[i]:+.3f}; {len(keep)} firms with nonzero weight; "
              f"self+top12 = {cum:.0%} of total |weight|")
        print(f"  {'ticker':7}{'weight':>9}  {'EV/EBITDA':>9}  subindustry / name")
        for j in top:
            print(f"  {tick[j]:7}{w[j]:+9.4f}  {mult[j]:9.1f}  "
                  f"{(subind[j] or '')[:26]:26} {(name[j] or '')[:24]}")

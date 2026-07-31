"""build_ml_frontier.py — ML-frontier (direct general-g LightGBM) peer-weight matrices for
the app, on the 2025 cross-section, BOTH ways:
  full  = single L2 GBM fit on all 2025 firms (1-fold) -> weight matrix S_full (has self-weights)
  cv    = 5-fold cross-fit (paper's stratified folds; a firm valued by the model trained on the
          other 4 folds) -> weight matrix S_cv. A firm's own fold (incl. itself) gets ZERO weight,
          so there is no self-weight, matching the paper's target-fold-pure discipline.

Our ML frontier is a LightGBM predicting t=log(EV/EBITDA) from 21 characteristics + GICS sub-
industry. A squared-error (L2) gradient-boosted prediction is EXACTLY a weighted average of the
TRAINING firms' target multiples (Geertsema-Lu App A.3): weight = accumulated leaf co-membership
across boosting iterations, scaled by the learning rate, normalized by leaf size + reg_lambda. We
use L2 (median-loss L1 is not a linear smoother, so it has no per-firm weights) and no bagging (so
each tree's leaf value is the mean over ALL its leaf-mates -> weights exact). Same folds as rung-1
(gk_2025_fold_assign.parquet). Everything else = the paper's frontier params.

Output (valuation_app/data/):
  ml_frontier.npz  S_full, S_cv (float32, n x n), tickers, multiple, fair_full, fair_cv, fold
  ml_frontier_leaves.npz  (LOCAL/reproducibility, gitignored): full-fit leaves + S_full + lr
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/Users/kerryback/repos/multiples/workspace")
PANEL = ROOT / "extend2000/data/panel_nonmicro.parquet"
ARTIFACT = ROOT / "logmult/artifact_data_2025.json"
FOLD_ASSIGN = ROOT / "logmult/results/gk_store/fold_membership.parquet"   # canonical folds (2025)
OUT = ROOT / "valuation_app/data"

SPLINE_FEATS = ["f_grossmargin", "f_ebitdamargin", "f_roa", "f_assetturn", "f_salegrow",
                "f_salegrow3", "f_assetgrow3", "f_ltg", "f_epsgrow", "has_ibes",
                "f_rdsale", "f_capxsale", "f_leverage", "f_cashassets", "f_tangibility",
                "f_accruals", "f_profitflag", "f_ebitdapos", "f_age", "log_at", "log_sale"]
CAT_FEAT = "gsubind_code"
LGB_FEATS = SPLINE_FEATS + [CAT_FEAT]
LR, LAMBDA, N_TREES, K, EPS = 0.05, 1.0, 400, 5, 1e-6
LGB = dict(objective="regression", n_estimators=N_TREES, learning_rate=LR, num_leaves=31,
           min_child_samples=50, subsample=1.0, colsample_bytree=1.0, reg_lambda=LAMBDA,
           random_state=42, n_jobs=-1, verbosity=-1, deterministic=True, force_row_wise=True)

# ---- 2025 cross-section ----
d = pd.read_parquet(PANEL)
d = d[(d.valyear == 2025) & (d.ebitda_yield > 0)].reset_index(drop=True).copy()
d["t"] = -np.log(d["ebitda_yield"]); d["multiple"] = 1.0 / d["ebitda_yield"]
d[CAT_FEAT] = d["gsubind"].astype("category").cat.codes.astype(int)
gvk = d["gvkey"].astype(str).to_numpy()
t = d["t"].to_numpy(float); n = len(d)
X = d[LGB_FEATS].copy(); X[CAT_FEAT] = X[CAT_FEAT].astype("category")

fa = pd.read_parquet(FOLD_ASSIGN)
fa = fa[fa["year"] == 2025].set_index("gvkey")["fold"]
fold = np.array([int(fa.loc[g]) for g in gvk])       # canonical 2025 folds (== rung-1)
firms = {r["gvkey"]: r for r in json.load(open(ARTIFACT))["firms"]}
tick = np.array([firms.get(g, {}).get("ticker") or "" for g in gvk], dtype=object)
mult = d["multiple"].to_numpy(float)
print(f"2025 firms N={n}; fold sizes={np.bincount(fold).tolist()}")


def leaf_mean_apply(leaves_row, R, lam):
    """A@R where A is the leaf-mean(+lam) operator over the ROWS indexing R (train firms).
    Returns per-leaf group means keyed for scatter."""
    grp = {}
    for lf in np.unique(leaves_row):
        idx = np.where(leaves_row == lf)[0]
        grp[lf] = R[idx, :].sum(0) / (len(idx) + lam)
    return grp


def smoother_full(leaves, lam):
    """Exact in-sample smoother S (n x n): raw = S @ t, rows sum to 1."""
    m = leaves.shape[0]
    B = np.full((m, m), 1.0 / m)
    S = B.copy(); R = np.eye(m) - B
    for j in range(leaves.shape[1]):
        grp = leaf_mean_apply(leaves[:, j], R, lam)
        AR = np.empty((m, m))
        for lf, g in grp.items():
            AR[leaves[:, j] == lf, :] = g
        S += LR * AR; R -= LR * AR
    return S


def smoother_cv_fold(leaves_tr, leaves_te, lam):
    """OOS smoother W (n_te x n_tr): raw_test = W @ t_train, over TRAIN firms only."""
    ntr = leaves_tr.shape[0]; nte = leaves_te.shape[0]
    W = np.full((nte, ntr), 1.0 / ntr)
    R = np.eye(ntr) - np.full((ntr, ntr), 1.0 / ntr)
    for j in range(leaves_tr.shape[1]):
        grp = leaf_mean_apply(leaves_tr[:, j], R, lam)          # keyed by leaf, over train rows
        AR_tr = np.empty((ntr, ntr)); AR_te = np.zeros((nte, ntr))
        lt, le = leaves_tr[:, j], leaves_te[:, j]
        for lf, g in grp.items():
            AR_tr[lt == lf, :] = g
            te_idx = np.where(le == lf)[0]
            if len(te_idx):
                AR_te[te_idx, :] = g
        W += LR * AR_te; R -= LR * AR_tr
    return W


# ---- FULL (1-fold) ----
m_full = lgb.LGBMRegressor(**LGB).fit(X, t, categorical_feature=[CAT_FEAT])
raw_full = m_full.predict(X, raw_score=True)
S_full = smoother_full(m_full.predict(X, pred_leaf=True), LAMBDA)
print(f"[full] reconstruction max|S t - raw| = {np.max(np.abs(S_full @ t - raw_full)):.2e}; "
      f"row-sum in [{S_full.sum(1).min():.4f},{S_full.sum(1).max():.4f}]")

# ---- CV (5-fold cross-fit) ----
S_cv = np.zeros((n, n)); raw_cv = np.full(n, np.nan)
for f in range(K):
    tr = np.where(fold != f)[0]; te = np.where(fold == f)[0]
    mf = lgb.LGBMRegressor(**LGB).fit(X.iloc[tr], t[tr], categorical_feature=[CAT_FEAT])
    raw_cv[te] = mf.predict(X.iloc[te], raw_score=True)
    W = smoother_cv_fold(mf.predict(X.iloc[tr], pred_leaf=True),
                         mf.predict(X.iloc[te], pred_leaf=True), LAMBDA)
    S_cv[np.ix_(te, tr)] = W        # test firms get weights over their training folds only
recon_cv = (S_cv @ t)
print(f"[cv]   reconstruction max|S_cv t - raw_cv| = {np.max(np.abs(recon_cv - raw_cv)):.2e}; "
      f"self-weight max |diag| = {np.max(np.abs(np.diag(S_cv))):.2e} (should be 0)")

np.savez_compressed(OUT / "ml_frontier.npz",
                    S_full=S_full.astype(np.float32), S_cv=S_cv.astype(np.float32),
                    tickers=tick.astype(str), multiple=mult.astype(np.float32),
                    fair_full=np.exp(raw_full).astype(np.float32),
                    fair_cv=np.exp(raw_cv).astype(np.float32),
                    fold=fold.astype(np.int8))
np.savez_compressed(OUT / "ml_frontier_leaves.npz", leaves=m_full.predict(X, pred_leaf=True),
                    S_full=S_full, S_cv=S_cv, learning_rate=LR, gvkeys=gvk, tickers=tick.astype(str),
                    fold=fold)
print("wrote ml_frontier.npz (S_full + S_cv) + ml_frontier_leaves.npz")

# ---- worked example: AAPL both ways ----
for ex in ["AAPL", "DPZ"]:
    i = np.where(tick == ex)[0]
    if not len(i):
        continue
    i = int(i[0])
    for lbl, Smat, fair in [("full", S_full, np.exp(raw_full)), ("cv", S_cv, np.exp(raw_cv))]:
        w = Smat[i]; nz = [j for j in np.argsort(-np.abs(w)) if j != i and abs(w[j]) >= EPS and tick[j]]
        print(f"{ex} [{lbl:4}] fold={fold[i]} fair {fair[i]:.1f}x  self {w[i]:+.3f}  "
              f"{len(nz)} nonzero; top: " + ", ".join(f"{tick[j]}={w[j]:+.3f}" for j in nz[:5]))

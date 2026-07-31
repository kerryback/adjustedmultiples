"""build_ml_frontier.py — ML-frontier (direct general-g LightGBM) peer-weight matrices for
the app, on the 2025 cross-section, for BOTH samples (non-micro and micro) and BOTH fits:
  full  = single L2 GBM fit on all of that sample's 2025 firms (1-fold) -> S_full (self-weights)
  cv    = 5-fold cross-fit (paper's stratified folds; a firm valued by the model trained on the
          other 4 folds) -> S_cv. A firm's own fold (incl. itself) gets ZERO weight, so there is
          no self-weight, matching the paper's target-fold-pure discipline.

The paper estimates each sample separately, so the micro frontier is trained only on micro firms
(its own folds) and the non-micro frontier only on non-micro firms. The app routes a ticker to the
matrices of its own sample.

Our ML frontier is a LightGBM predicting t=log(EV/EBITDA) from 21 characteristics + GICS sub-
industry. A squared-error (L2) gradient-boosted prediction is EXACTLY a weighted average of the
TRAINING firms' target multiples (Geertsema-Lu App A.3): weight = accumulated leaf co-membership
across boosting iterations, scaled by the learning rate, normalized by leaf size + reg_lambda. We
use L2 (median-loss L1 is not a linear smoother, so it has no per-firm weights) and no bagging (so
each tree's leaf value is the mean over ALL its leaf-mates -> weights exact). Same folds as the
adjusted-multiple curves (gk_store fold membership). Everything else = the paper's frontier params.

Output (valuation_app/data/ml_frontier.npz), per sample s in {nonmicro, micro}:
  S_full_{s}, S_cv_{s} (float32, n_s x n_s), tickers_{s}, multiple_{s}, fair_full_{s},
  fair_cv_{s}, fold_{s}.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/Users/kerryback/repos/multiples/workspace")
ARTIFACT = ROOT / "logmult/artifact_data_2025.json"
STORE = ROOT / "logmult/results/gk_store"
OUT = ROOT / "valuation_app/data"
SAMPLES = {
    "nonmicro": (ROOT / "extend2000/data/panel_nonmicro.parquet", STORE / "fold_membership.parquet"),
    "micro":    (ROOT / "extend2000/data/panel_micro.parquet",    STORE / "fold_membership_micro.parquet"),
}

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

firms_meta = {r["gvkey"]: r for r in json.load(open(ARTIFACT))["firms"]}


def leaf_mean_apply(leaves_row, R, lam):
    """A@R where A is the leaf-mean(+lam) operator over the ROWS indexing R (train firms)."""
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
        grp = leaf_mean_apply(leaves_tr[:, j], R, lam)
        AR_tr = np.empty((ntr, ntr)); AR_te = np.zeros((nte, ntr))
        lt, le = leaves_tr[:, j], leaves_te[:, j]
        for lf, g in grp.items():
            AR_tr[lt == lf, :] = g
            te_idx = np.where(le == lf)[0]
            if len(te_idx):
                AR_te[te_idx, :] = g
        W += LR * AR_te; R -= LR * AR_tr
    return W


def build_sample(sample, panel_path, fold_path):
    d = pd.read_parquet(panel_path)
    d = d[(d.valyear == 2025) & (d.ebitda_yield > 0)].reset_index(drop=True).copy()
    d["t"] = -np.log(d["ebitda_yield"]); d["multiple"] = 1.0 / d["ebitda_yield"]
    d[CAT_FEAT] = d["gsubind"].astype("category").cat.codes.astype(int)
    gvk = d["gvkey"].astype(str).to_numpy()
    t = d["t"].to_numpy(float); n = len(d)
    X = d[LGB_FEATS].copy(); X[CAT_FEAT] = X[CAT_FEAT].astype("category")

    fa = pd.read_parquet(fold_path)
    fa = fa[fa["year"] == 2025].set_index("gvkey")["fold"]
    fold = np.array([int(fa.loc[g]) for g in gvk])
    tick = np.array([firms_meta.get(g, {}).get("ticker") or "" for g in gvk], dtype=object)
    mult = d["multiple"].to_numpy(float)
    print(f"[{sample}] 2025 firms N={n}; fold sizes={np.bincount(fold).tolist()}")

    # FULL (1-fold)
    m_full = lgb.LGBMRegressor(**LGB).fit(X, t, categorical_feature=[CAT_FEAT])
    raw_full = m_full.predict(X, raw_score=True)
    S_full = smoother_full(m_full.predict(X, pred_leaf=True), LAMBDA)
    print(f"  [full] max|S t - raw| = {np.max(np.abs(S_full @ t - raw_full)):.2e}; "
          f"row-sum in [{S_full.sum(1).min():.4f},{S_full.sum(1).max():.4f}]")

    # CV (5-fold cross-fit)
    S_cv = np.zeros((n, n)); raw_cv = np.full(n, np.nan)
    for f in range(K):
        tr = np.where(fold != f)[0]; te = np.where(fold == f)[0]
        mf = lgb.LGBMRegressor(**LGB).fit(X.iloc[tr], t[tr], categorical_feature=[CAT_FEAT])
        raw_cv[te] = mf.predict(X.iloc[te], raw_score=True)
        W = smoother_cv_fold(mf.predict(X.iloc[tr], pred_leaf=True),
                             mf.predict(X.iloc[te], pred_leaf=True), LAMBDA)
        S_cv[np.ix_(te, tr)] = W
    print(f"  [cv]   max|S_cv t - raw_cv| = {np.max(np.abs((S_cv @ t) - raw_cv)):.2e}; "
          f"self-weight max |diag| = {np.max(np.abs(np.diag(S_cv))):.2e} (should be 0)")

    return {
        f"S_full_{sample}": S_full.astype(np.float32), f"S_cv_{sample}": S_cv.astype(np.float32),
        f"tickers_{sample}": tick.astype(str), f"multiple_{sample}": mult.astype(np.float32),
        f"fair_full_{sample}": np.exp(raw_full).astype(np.float32),
        f"fair_cv_{sample}": np.exp(raw_cv).astype(np.float32), f"fold_{sample}": fold.astype(np.int8),
    }


payload = {"samples": np.array(list(SAMPLES.keys()), dtype=str)}
for s, (pp, fp) in SAMPLES.items():
    payload.update(build_sample(s, pp, fp))
np.savez_compressed(OUT / "ml_frontier.npz", **payload)
print(f"wrote {OUT/'ml_frontier.npz'} for samples {list(SAMPLES.keys())}")

# worked examples
for ex in ["AAPL", "DPZ", "ACU", "BKTI"]:
    for s in SAMPLES:
        tk = payload[f"tickers_{s}"]; i = np.where(tk == ex)[0]
        if not len(i):
            continue
        i = int(i[0]); Sf = payload[f"S_full_{s}"]; ff = payload[f"fair_full_{s}"]
        w = Sf[i]; nz = [j for j in np.argsort(-np.abs(w)) if j != i and abs(w[j]) >= EPS and tk[j]]
        print(f"{ex} [{s}/full] fair {ff[i]:.1f}x  self {w[i]:+.3f}  {len(nz)} nonzero; "
              + ", ".join(f"{tk[j]}={w[j]:+.3f}" for j in nz[:4]))

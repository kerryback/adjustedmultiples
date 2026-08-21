# adjustedmultiples

Value a public firm by *adjusted multiples*: take its peers, adjust each peer's
EV/EBITDA multiple for how it differs from the target on operating characteristics, and
take the median of the adjusted peer multiples as the fair multiple. This is the
interpretable (rung-1) estimator from the *adjusted multiple* research project, served as
a small FastAPI web app.

For a target firm *i* and each peer *j*:

```
adjusted multiple of peer j  =  m_j · exp( Σ_k [ g_k(x_i) − g_k(x_j) ] )
fair multiple of i           =  median over peers of the adjusted peer multiples
fair enterprise value        =  fair multiple · EBITDA_i
```

The median is taken in logs, where the estimator takes it. On an odd-sized peer set that
is the same number; on an even-sized one it averages the two middle multiples
geometrically rather than arithmetically, and the two answers differ by enough to see.

## Peers — Rule D

A firm's peers are the five to ten closest firms the matching ladder reaches: same GICS
sector, widening outward through sub-industry, industry and industry group, and from a
factor of two on lagged market cap out to a factor of five, stopping at the first step
that finds five candidates.

Under **Rule D** (`method_spec.md` sec. 13d, adopted 2026-08-20) the ladder draws those
candidates from the **whole** April-30 cross-section rather than from the target's own
size universe, so a microcap can be a peer of a large firm and the other way round. The
app marks such a peer `cross`. The curves stay per universe: a target is always rated by
its own universe's `g`, whichever universe each peer came from.

## Curves

The `g_k` are the additive appraisal curves of the Rule D lead cell — the production
per-fold coefficient vectors, not a refit. The bundle carries every additive piece of
`g`: exact piecewise cubics per characteristic, the flag effects, the GICS sub-industry
class effects, and the value the design pipeline gives a *missing* characteristic. So the
app computes the estimator rather than an approximation of it, and `check_bundle.py`
asserts exactly that — it values every 2025 firm through the app's own code path and
compares against the paper's stored prediction.

Two fits are offered, from the *Value firm* menu. The 5-fold cross-fit is the paper's:
each firm is valued out-of-sample by the model fitted without its own fold. The
full-sample fit is a display object, fitted for this site alone on all 2025 rows under
the same Rule D peer sets, the same inherited η\*, and the same five backfit alternations.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
# open http://127.0.0.1:8000
```

Enter a ticker → the firm, its fair vs market valuation, and its peer set with each
peer's adjusted multiple. Pick a peer from the dropdown to see the adjustment built up
characteristic by characteristic.

## Layout

- `app.py` — FastAPI backend and the valuation logic.
- `static/index.html` — the single-page frontend.
- `data/app_data.json` — pre-built bundle (2025 firms, characteristics, Rule D peer sets,
  and the fitted curve sets). This is what the app serves; no research data or pandas at
  runtime.
- `data/ml_frontier.npz` — the machine-learning frontier's implicit peer weights, for the
  Gradient Boosted tab. The frontier has no peer step, so the peer rule does not move it
  and it was not rebuilt.
- `prep_data.py` — dev tool that assembles `data/app_data.json` from the research repo's
  `workspaces/kerry-back/analyst/ruled/code/16_site_ruled.py` output. Not used at runtime.
- `check_bundle.py` — dev tool that checks the assembled bundle against the paper's
  stored numbers. Run it before deploying.
- `Dockerfile` — container image (used by the Koyeb deploy).

## Deploy

Built and deployed on [Koyeb](https://koyeb.com) from this GitHub repo (Docker build),
served at **adjustedmultiples.com**.

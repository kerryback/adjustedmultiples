# adjustedmultiples

Value a public firm by **adjusted multiples**: take its size-and-industry peers,
adjust each peer's EV/EBITDA multiple for how it differs from the target on operating
characteristics, and take the median of the adjusted peer multiples as the fair
multiple. This is the interpretable (rung-1) estimator from the *adjusted multiple*
research project, served as a small FastAPI web app.

For a target firm *i* and each peer *j*:

```
adjusted multiple of peer j  =  m_j · exp( Σ_k [ g_k(x_i) − g_k(x_j) ] )
fair multiple of i           =  median over peers of the adjusted peer multiples
fair enterprise value        =  fair multiple · EBITDA_i
```

The `g_k` are the additive appraisal curves (one per operating characteristic), fit
on 2001–2025 positive-EBITDA U.S. firms. Microcaps and non-microcaps use **separate**
curve sets and are matched to peers **within their own universe**; the app picks the
curve set from the target firm's size class.

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
- `data/app_data.json` — pre-built bundle (2025 firms, characteristics, peer sets, and
  the per-sample `g_k` curves). This is what the app serves; no research data or pandas
  at runtime.
- `prep_data.py` — dev tool that regenerates `data/app_data.json` from the research
  workspace. Not used at runtime.
- `Dockerfile` — container image (used by the Koyeb deploy).

## Deploy

Built and deployed on [Koyeb](https://koyeb.com) from this GitHub repo (Docker build),
served at **adjustedmultiples.com**.

## Note

The `g_k` curves are being refreshed on the corrected research panel; the microcap
curve set is currently a placeholder (`micro_curves_placeholder` in the data bundle)
until the microcap fit is regenerated. The valuation logic and structure are final.

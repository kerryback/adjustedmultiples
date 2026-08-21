"""prep_data.py — assemble the valuation app's data bundle.

Reads the two files the research repo's Rule D site builder writes and merges
them into `data/app_data.json`, which `app.py` loads at startup:

  site_firms_2025.json    one record per 2025 firm — identity, characteristics,
                          EBITDA, EV, market cap, cross-fit fold, and its RULE D
                          peer set
  site_curves_2025.json   per sample, six fitted curve sets (five fold-out, one
                          full-sample), each as exact piecewise cubics plus the
                          flag effects, the gsubind class effects, and the value
                          the design pipeline gives a missing characteristic

Both are produced by, and only by,

  workspaces/kerry-back/analyst/ruled/code/16_site_ruled.py

in the `multiples` repo, whose two gates assert that the emitted bundle
reproduces the paper's stored Rule D prediction for every firm in both samples.
That is the reason this script does no estimation and reads no panel: any
arithmetic done here would be arithmetic those gates did not check. It merges
and writes, nothing else.

The peer rule is RULE D (method_spec.md sec. 13d): the ladder is unchanged, but
a firm's candidate peers are drawn from the whole April-30 cross-section rather
than from its own size universe, so a microcap can now be a peer of a large firm
and the other way round. The curves are the Rule D lead cell's own fits.

Usage:  python prep_data.py            (reads ~/repos/multiples, or $MULTIPLES_REPO)
"""
import json
import os
from pathlib import Path

MULT = Path(os.environ.get("MULTIPLES_REPO",
                           Path.home() / "repos" / "multiples"))
SITE = MULT / "workspaces/kerry-back/analyst/ruled/results/site"
FIRMS_IN = SITE / "site_firms_2025.json"
CURVES_IN = SITE / "site_curves_2025.json"

OUT = Path(__file__).resolve().parent / "data"


def main():
    for p in (FIRMS_IN, CURVES_IN):
        if not p.exists():
            raise SystemExit(f"missing {p}\nRun 16_site_ruled.py in {MULT} first.")
    firms = json.loads(FIRMS_IN.read_text())
    curves = json.loads(CURVES_IN.read_text())

    assert firms["year"] == curves["year"] == 2025
    assert firms["peer_rule"] == curves["peer_rule"] == "D", "not a Rule D build"
    assert firms["target"] == curves["target"] == "logmult"

    # every peer must be a firm the bundle can display, or the site quietly
    # shows a shorter peer set than the estimator used
    have = {f["gvkey"] for f in firms["firms"]}
    dangling = sum(len([g for g in f["peers"] if g not in have])
                   for f in firms["firms"])
    assert dangling == 0, f"{dangling} peer slots point outside the bundle"

    bundle = {
        "year": 2025,
        "peer_rule": "D",
        "features": firms["features"],
        "n_folds": curves["n_folds"],
        "curves": curves["curves"],
        "firms": firms["firms"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "app_data.json"
    dest.write_text(json.dumps(bundle))

    for samp in bundle["curves"]:
        fs = [f for f in bundle["firms"] if f["sample"] == samp]
        npeers = sorted(len(f["peers"]) for f in fs)
        out_of = sum(1 for f in fs
                     if any(next(x for x in bundle["firms"] if x["gvkey"] == g)
                            ["sample"] != samp for g in f["peers"]))
        print(f"[prep] {samp}: {len(fs)} firms, median peers "
              f"{npeers[len(npeers)//2]}, {out_of} with an out-of-sample peer, "
              f"full + {len(bundle['curves'][samp]['folds'])} fold curve sets")
    print(f"[prep] wrote {dest} ({dest.stat().st_size/1e6:.1f} MB, "
          f"{len(bundle['firms'])} firms)")
    for tk in ("AAPL", "CALM", "DPZ", "HPQ"):
        m = next((f for f in bundle["firms"] if f["ticker"] == tk), None)
        if m:
            print(f"  {tk}: {m['name']} mult={m['multiple']} "
                  f"peers={len(m['peers'])} sample={m['sample']}")


if __name__ == "__main__":
    main()

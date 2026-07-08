"""
semantic_core.py — semantic-distance test for the competitor, with a frequency-matched same-relation
null. The scientific claim is NOT "wrong answers are semantically close"; it is "the SELECTED
alternative is closer to the gold answer than frequency-matched same-relation decoys are" -- i.e.
closer than relation-type + frequency alone would predict. Pure numpy; embeddings/ontology lookups
are supplied by the notebook.
"""
import numpy as np

def cosine(a, b):
    a=np.asarray(a,np.float64); b=np.asarray(b,np.float64)
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb)) if na>0 and nb>0 else 0.0

def freq_matched_decoys(gold_id, selected_id, pool_ids, freq, k=20, tol=1.0, rng=None):
    """Pick up to k same-relation decoys whose token log-frequency is within `tol` of the SELECTED
    alternative's, excluding the gold and the selected token. Matching to the selected (not the gold)
    controls the null at the frequency where the competition actually happened."""
    rng=rng or np.random.default_rng(0)
    fsel=freq[selected_id]
    cand=[p for p in pool_ids if p!=gold_id and p!=selected_id and abs(freq[p]-fsel)<=tol]
    if len(cand)>k: cand=list(rng.choice(cand,k,replace=False))
    return cand

def matched_null_test(sim_selected, sims_null):
    """Given the gold<->selected similarity and a list of gold<->decoy similarities (the
    frequency-matched same-relation null), return the percentile of the selected similarity within the
    null and the z-score. A high percentile means the selected alternative is unusually close to the
    gold beyond type+frequency."""
    n=np.asarray(sims_null,np.float64)
    if n.size==0: return dict(percentile=float("nan"), z=float("nan"), delta=float("nan"), n_null=0)
    pct=float((n<sim_selected).mean())
    mu=float(n.mean()); sd=float(n.std())
    z=float((sim_selected-mu)/sd) if sd>0 else float("nan")
    return dict(percentile=pct, z=z, delta=float(sim_selected-mu), n_null=int(n.size))

def aggregate_null_tests(results):
    """Aggregate per-item matched-null tests. Under the null (selected no closer than decoys) the mean
    percentile is 0.5; >0.5 means the selected alternative is systematically closer to the gold.
    Reports the mean percentile, a one-sample test that it exceeds 0.5, and the mean delta/z."""
    pcts=np.array([r["percentile"] for r in results if np.isfinite(r["percentile"])])
    deltas=np.array([r["delta"] for r in results if np.isfinite(r["delta"])])
    zs=np.array([r["z"] for r in results if np.isfinite(r["z"])])
    if pcts.size==0: return dict(n=0)
    # Wilcoxon signed-rank of (percentile - 0.5)
    w_p=_wilcoxon_p(pcts-0.5)
    # bootstrap CI on mean percentile
    rng=np.random.default_rng(0)
    bs=np.array([pcts[rng.integers(0,len(pcts),len(pcts))].mean() for _ in range(2000)])
    return dict(n=int(pcts.size), mean_percentile=float(pcts.mean()),
                ci95_low=float(np.percentile(bs,2.5)), ci95_high=float(np.percentile(bs,97.5)),
                frac_above_half=float((pcts>0.5).mean()),
                mean_delta=float(deltas.mean()) if deltas.size else float("nan"),
                mean_z=float(zs.mean()) if zs.size else float("nan"),
                wilcoxon_p_gt_half=w_p)

def _wilcoxon_p(d):
    """Two-sided Wilcoxon signed-rank p-value for median(d)=0 (normal approx, ties/zeros dropped)."""
    d=np.asarray(d,np.float64); d=d[d!=0]
    n=d.size
    if n<6: return float("nan")
    r=_rankdata(np.abs(d)); Wp=float(r[d>0].sum()); Wm=float(r[d<0].sum())
    W=min(Wp,Wm); mu=n*(n+1)/4.0; sd=np.sqrt(n*(n+1)*(2*n+1)/24.0)
    from math import erf, sqrt
    z=(W-mu)/sd
    return float(2*0.5*(1+erf(-abs(z)/sqrt(2))))

def _rankdata(a):
    a=np.asarray(a,np.float64); o=np.argsort(a); r=np.empty(len(a),float); i=0
    while i<len(a):
        j=i
        while j+1<len(a) and a[o[j+1]]==a[o[i]]: j+=1
        r[o[i:j+1]]=(i+j)/2.0+1.0; i=j+1
    return r

"""
baseline_core.py — exact baseline/contextual decomposition of the final-readout margin.

z_i(v) = W_U[v] . h_i. Decompose h_i = hbar + c_i, where hbar is a HELD-OUT (leave-one-out) mean of
the final residual across items, and c_i is the item-specific deviation. Then:
    b_v       = W_U[v] . hbar          (context-independent readout baseline for token v)
    c_i(v)    = W_U[v] . c_i           (item-specific contextual term)
    M_i(a,b)  = z_i(a)-z_i(b) = (b_a - b_b) + (c_i(a) - c_i(b))
              =  baseline_margin       +  contextual_margin     (EXACT)

The leave-one-out mean ensures item i does not contribute to its own baseline. Pure numpy.
"""
import numpy as np

def loo_mean(H):
    """Leave-one-out mean of rows of H: loo[i] = (sum_j H[j] - H[i]) / (n-1)."""
    H=np.asarray(H,np.float64); n=H.shape[0]
    if n<2: return np.zeros_like(H)
    tot=H.sum(0)
    return (tot[None,:]-H)/(n-1)

def decompose_margin(Wu, H, ans_ids, comp_ids, use_loo=True):
    """For each item i with answer a_i and competitor b_i, return the exact baseline/contextual/final
    margin decomposition using a leave-one-out (or global) mean residual.
    Wu: (V,d). H: (n,d) final residuals. ans_ids,comp_ids: length-n token ids."""
    Wu=np.asarray(Wu,np.float64); H=np.asarray(H,np.float64); n=H.shape[0]
    hbar = loo_mean(H) if use_loo else np.repeat(H.mean(0)[None,:],n,axis=0)
    C = H - hbar                                  # contextual deviations (n,d)
    rows=[]
    for i in range(n):
        a=int(ans_ids[i]); b=int(comp_ids[i])
        b_a=float(Wu[a]@hbar[i]); b_b=float(Wu[b]@hbar[i])
        c_a=float(Wu[a]@C[i]);    c_b=float(Wu[b]@C[i])
        base_m=b_a-b_b; ctx_m=c_a-c_b; final_m=base_m+ctx_m
        # exactness check: final_m should equal z_i(a)-z_i(b)
        z_a=float(Wu[a]@H[i]); z_b=float(Wu[b]@H[i])
        rows.append(dict(baseline_margin=base_m, contextual_margin=ctx_m, final_margin=final_m,
                         exact_residual=float(final_m-(z_a-z_b)),
                         b_ans=b_a, b_comp=b_b, c_ans=c_a, c_comp=c_b))
    return rows

def regime_label(base_m, ctx_m):
    """Two regimes (no 'competition-limited' metaphor):
    - answer-support limited: the answer's contextual signal is the shortfall; ctx margin is the more
      negative term and baseline is not strongly against the answer.
    - baseline/alternative-support limited: the baseline margin is strongly negative (the selected
      alternative has too much readout baseline), so answer-up alone is likely insufficient."""
    if base_m < 0 and base_m <= ctx_m:
        return "baseline_limited"
    return "answer_support_limited"

def aggregate(rows):
    bm=np.array([r["baseline_margin"] for r in rows]); cm=np.array([r["contextual_margin"] for r in rows])
    fm=np.array([r["final_margin"] for r in rows])
    labels=[regime_label(r["baseline_margin"],r["contextual_margin"]) for r in rows]
    return dict(n=len(rows),
                mean_baseline_margin=float(bm.mean()), mean_contextual_margin=float(cm.mean()),
                mean_final_margin=float(fm.mean()),
                frac_baseline_negative=float((bm<0).mean()),
                frac_baseline_limited=float(np.mean([l=="baseline_limited" for l in labels])),
                max_abs_exactness=float(np.max(np.abs([r["exact_residual"] for r in rows]))))

# ---------- alignment tests: where does the baseline come from? ----------
def frequency_direction(Wu, freq):
    f=np.asarray(freq,np.float64); fbar=f.mean()
    r=((f-fbar)[:,None]*np.asarray(Wu,np.float64)).sum(0); n=np.linalg.norm(r)
    return r/n if n>0 else r

def baseline_alignment(Wu, hbar_global, freq, bos_dir=None):
    """Tests for how the baseline becomes operational:
      cos(mean residual, frequency direction)
      corr(b_v, token frequency)               where b_v = W_U[v].hbar
      corr(b_v, W_U[v].r)                       (baseline vs unembedding frequency projection)
      cos(mean residual, BOS/sink direction)    if bos_dir provided
    """
    Wu=np.asarray(Wu,np.float64); hbar=np.asarray(hbar_global,np.float64); f=np.asarray(freq,np.float64)
    r=frequency_direction(Wu,freq)
    def cos(a,b):
        na=np.linalg.norm(a); nb=np.linalg.norm(b); return float(a@b/(na*nb)) if na>0 and nb>0 else 0.0
    b_v=Wu@hbar
    proj_r=Wu@r
    out=dict(cos_meanres_freqdir=cos(hbar,r),
             corr_baseline_freq=float(np.corrcoef(b_v,f)[0,1]) if b_v.std()>0 else float("nan"),
             corr_baseline_unembedfreqproj=float(np.corrcoef(b_v,proj_r)[0,1]) if b_v.std()>0 and proj_r.std()>0 else float("nan"))
    if bos_dir is not None:
        out["cos_meanres_bosdir"]=cos(hbar,np.asarray(bos_dir,np.float64))
    return out

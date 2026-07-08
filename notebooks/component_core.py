"""
component_core.py — component-level margin attribution and its baseline/contextual split.

The final residual is a sum of component outputs (embeddings + each layer's attention block + each
layer's MLP block), all in the residual stream:
    h_i = sum_k x_{i,k}
For a failed item with answer a_i and selected alternative b_i, each component's contribution to the
answer-vs-alternative margin is
    dz_{i,k} = < x_{i,k}, W_U[a_i] - W_U[b_i] >            (through the final norm; see runner)
Splitting each component into a baseline (mean) and contextual (deviation) part, x_{i,k} = xbar_k + dx_{i,k}:
    dz^base_{i,k} = < xbar_k,    W_U[a_i]-W_U[b_i] >
    dz^ctx_{i,k}  = < dx_{i,k},  W_U[a_i]-W_U[b_i] >
xbar_k is a leave-one-out mean (item i excluded). Pure numpy; the runner supplies component outputs.
"""
import numpy as np

def loo_mean_stack(X):
    """X: (n, K, d) component outputs. Returns LOO mean over items for each component: (n,K,d)."""
    X=np.asarray(X,np.float64); n=X.shape[0]
    if n<2: return np.zeros_like(X)
    tot=X.sum(0)                       # (K,d)
    return (tot[None]-X)/(n-1)

def attribute_margins(X, Wu, ans_ids, comp_ids, use_loo=True):
    """X: (n,K,d) per-component residual contributions (already projected through final norm).
    Returns per-component total / baseline / contextual margin contributions, averaged over items,
    plus per-item arrays for ranking. dz total per component should sum (over k) to the final margin."""
    X=np.asarray(X,np.float64); Wu=np.asarray(Wu,np.float64); n,K,d=X.shape
    xbar = loo_mean_stack(X) if use_loo else np.repeat(X.mean(0)[None],n,axis=0)
    dX = X - xbar
    tot=np.zeros((n,K)); base=np.zeros((n,K)); ctx=np.zeros((n,K))
    for i in range(n):
        diff=Wu[int(ans_ids[i])]-Wu[int(comp_ids[i])]   # (d,)
        tot[i]=X[i]@diff
        base[i]=xbar[i]@diff
        ctx[i]=dX[i]@diff
    return dict(total=tot, baseline=base, contextual=ctx,
                mean_total=tot.mean(0), mean_baseline=base.mean(0), mean_contextual=ctx.mean(0))

def rank_components(mean_vec, names, k=10, most_negative=True):
    """Return the top-k components by mean contribution. most_negative=True ranks components that push
    toward the SELECTED ALTERNATIVE (negative answer-minus-alternative margin)."""
    idx=np.argsort(mean_vec)            # ascending: most negative first
    if not most_negative: idx=idx[::-1]
    return [(names[i], float(mean_vec[i])) for i in idx[:k]]

def margin_shift(delta_z_before, delta_z_after):
    """Mean change in the answer-vs-alternative margin after an intervention (positive = toward gold)."""
    a=np.asarray(delta_z_before,float); b=np.asarray(delta_z_after,float)
    return dict(mean_before=float(a.mean()), mean_after=float(b.mean()),
                mean_shift=float((b-a).mean()), frac_improved=float((b>a).mean()))

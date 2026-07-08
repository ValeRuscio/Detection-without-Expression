"""
semreg_core.py — regression/predictor analysis for what gets selected as the competitor, plus
external-semantics-vs-model-geometry comparison. Builds on semantic_core's matched null.

Two regression framings:
  (A) SELECTION model (primary, non-circular): among same-relation candidates for an item, which one
      is selected? Per-item softmax/conditional-logit over candidate features. Asks whether semantic
      similarity raises selection odds after controlling for frequency, type, and unembedding cosine.
  (B) MARGIN model (secondary, note circularity): z_b - z_a regressed on the same predictors. The
      selected b is chosen by the logit, so features of b correlate mechanically with the margin;
      reported only as a descriptive cross-check, not the headline.

Pure numpy + a small IRLS logistic fitter (no sklearn dependency, though sklearn may be used in the
notebook). Standardization is applied to predictors for comparable coefficients.
"""
import numpy as np

FEATURES=["d_logfreq","same_type","sem_sim","unembed_cos","final_logit"]

def standardize(X):
    X=np.asarray(X,np.float64); mu=X.mean(0); sd=X.std(0); sd[sd==0]=1.0
    return (X-mu)/sd, mu, sd

# ---------- logistic regression via IRLS (binary), with L2 ridge for stability ----------
def logistic_fit(X, y, l2=1.0, iters=50):
    X=np.asarray(X,np.float64); y=np.asarray(y,np.float64)
    n,d=X.shape; Xb=np.hstack([np.ones((n,1)),X]); w=np.zeros(d+1)
    R=np.eye(d+1)*l2; R[0,0]=0.0
    for _ in range(iters):
        eta=Xb@w; p=1/(1+np.exp(-np.clip(eta,-30,30))); W=np.clip(p*(1-p),1e-6,None)
        z=eta+(y-p)/W
        A=Xb.T@(Xb*W[:,None])+R; b=Xb.T@(W*z)
        w_new=np.linalg.solve(A,b)
        if np.max(np.abs(w_new-w))<1e-8: w=w_new; break
        w=w_new
    return w  # [intercept, coef_per_feature...]

def logistic_se(X, w, l2=1.0):
    """Approx standard errors from the inverse Fisher information (ridge-penalized)."""
    X=np.asarray(X,np.float64); n,d=X.shape; Xb=np.hstack([np.ones((n,1)),X])
    eta=Xb@w; p=1/(1+np.exp(-np.clip(eta,-30,30))); W=np.clip(p*(1-p),1e-6,None)
    R=np.eye(d+1)*l2; R[0,0]=0.0
    cov=np.linalg.inv(Xb.T@(Xb*W[:,None])+R)
    return np.sqrt(np.clip(np.diag(cov),0,None))

def wald_p(coef, se):
    from math import erf,sqrt
    z=coef/se if se>0 else 0.0
    return float(2*0.5*(1+erf(-abs(z)/sqrt(2))))

# ---------- (A) selection model: per-item candidate rows, label = is this the selected winner ----------
def build_selection_rows(items_feats, features=None):
    features=features or FEATURES
    """items_feats: list over items, each a dict with 'cands' = list of per-candidate feature dicts
    (keys in FEATURES) and 'winner_idx' = index of the selected candidate. Returns stacked X
    (standardized within the full design), y, and group ids for per-item normalization."""
    X=[]; y=[]; grp=[]
    for gi,it in enumerate(items_feats):
        for ci,c in enumerate(it["cands"]):
            X.append([c[f] for f in features]); y.append(int(ci==it["winner_idx"])); grp.append(gi)
    X=np.asarray(X,np.float64); y=np.asarray(y,int); grp=np.asarray(grp,int)
    return X,y,grp

def fit_selection(items_feats, l2=1.0, features=None):
    features=features or FEATURES
    """Fit the selection logistic over all candidate rows (within-item competition approximated by
    pooled logistic with standardized features). Returns standardized coefficients, SEs, p-values."""
    X,y,grp=build_selection_rows(items_feats, features)
    if len(y)<20 or y.sum()==0: return dict(n=len(y), status="insufficient")
    Xs,mu,sd=standardize(X)
    w=logistic_fit(Xs,y,l2=l2); se=logistic_se(Xs,w,l2=l2)
    out=dict(n_rows=int(len(y)), n_items=int(grp.max()+1), intercept=float(w[0]))
    for i,f in enumerate(features):
        out[f]=dict(coef=float(w[i+1]), se=float(se[i+1]), p=wald_p(w[i+1],se[i+1]))
    return out

# ---------- (B) margin model (descriptive; circularity noted) ----------
def fit_margin(margins, feats):
    """OLS of (z_b - z_a) on standardized predictors. Returns standardized betas, SEs, p. Descriptive
    only: b is selected by the logit, so this is not a clean causal control."""
    Y=np.asarray(margins,np.float64); X=np.asarray([[f[k] for k in FEATURES] for f in feats],np.float64)
    if len(Y)<20: return dict(n=len(Y), status="insufficient")
    Xs,mu,sd=standardize(X); Xb=np.hstack([np.ones((len(Y),1)),Xs])
    beta,_,_,_=np.linalg.lstsq(Xb,Y,rcond=None)
    resid=Y-Xb@beta; dof=max(len(Y)-Xb.shape[1],1); s2=float(resid@resid/dof)
    cov=s2*np.linalg.pinv(Xb.T@Xb); se=np.sqrt(np.clip(np.diag(cov),0,None))
    out=dict(n=int(len(Y)), r2=float(1-(resid@resid)/(((Y-Y.mean())**2).sum()+1e-12)), intercept=float(beta[0]))
    for i,f in enumerate(FEATURES):
        out[f]=dict(beta=float(beta[i+1]), se=float(se[i+1]), p=wald_p(beta[i+1],se[i+1]))
    return out

# ---------- (3) external semantics vs model geometry ----------
def external_vs_internal(sem_sims, unembed_coss):
    """Correlate external semantic similarity (WordNet/embedding) with model-internal unembedding
    cosine, over (gold, selected) pairs. High correlation links the semantic story to output geometry."""
    a=np.asarray(sem_sims,np.float64); b=np.asarray(unembed_coss,np.float64)
    m=np.isfinite(a)&np.isfinite(b); a,b=a[m],b[m]
    if len(a)<5: return dict(n=int(len(a)), status="insufficient")
    # Pearson + Spearman
    pear=float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else float("nan")
    import mech_core as _mc
    return dict(n=int(len(a)), pearson=pear, spearman=_mc.spearman(a,b),
                mean_sem=float(a.mean()), mean_unembed=float(b.mean()))

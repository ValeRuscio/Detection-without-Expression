"""
tunedlens_core.py — tuned-lens probe fitting and read scoring.

A tuned lens replaces the raw logit lens (LN_f(h_l) @ W_U) with a per-layer affine probe applied
BEFORE the shared unembedding:
    logits_l = ( A_l h_l + b_l ) @ W_U^T            (W_U fixed; A_l in R^{d x d}, b_l in R^d)
A_l, b_l are fit per layer to match the model's FINAL logits (distillation), on HELD-OUT activations so
the read used to measure the dissociation is not circular. Closed-form ridge regression in logit space
is unstable (V is huge); instead we fit A_l, b_l to map h_l -> h_final (the final residual) by ridge
regression (translator form of the tuned lens), then read through the true final norm + unembedding.
This is the standard "tuned lens as a learned translator to the final layer" formulation. Pure numpy.
"""
import numpy as np

def fit_translator(Hl, Hfinal, l2=1.0):
    """Ridge-fit A,b so that A h_l + b ~= h_final. Hl,(Hfinal): (n,d). Returns (A (d,d), b (d,))."""
    Hl=np.asarray(Hl,np.float64); Hf=np.asarray(Hfinal,np.float64); n,d=Hl.shape
    X=np.hstack([Hl,np.ones((n,1))])                 # (n,d+1)
    R=np.eye(d+1)*l2; R[-1,-1]=0.0
    W=np.linalg.solve(X.T@X+R, X.T@Hf)               # (d+1,d)
    A=W[:d].T                                         # (d,d): h_final ~= A h_l + b
    b=W[d]
    return A,b

def translator_read(h_l, A, b):
    """Apply the fitted translator: returns the predicted final-layer residual A h_l + b."""
    return A@np.asarray(h_l,np.float64)+b

def fit_all_layers(H_by_layer, Hfinal, layers, l2=1.0):
    """Fit a translator per layer in `layers`. H_by_layer: dict layer-> (n,d). Returns dict layer->(A,b)."""
    return {l: fit_translator(H_by_layer[l], Hfinal, l2=l2) for l in layers}

def tuned_logprob(h_l, A, b, Wu, final_norm_fn):
    """Tuned-lens log-probabilities: translate h_l to the final residual, apply the TRUE final norm and
    unembedding. final_norm_fn maps a (d,) residual to the normed (d,) vector."""
    hf=translator_read(h_l,A,b)
    hn=final_norm_fn(hf)
    z=Wu@hn
    z=z-z.max(); lp=z-np.log(np.exp(z).sum())
    return lp

def reconstruction_quality(H_by_layer, Hfinal, probes, layers):
    """Per-layer R^2 of the translator (how well A h_l + b predicts h_final). Sanity that probes fit."""
    out={}
    Hf=np.asarray(Hfinal,np.float64); ss_tot=((Hf-Hf.mean(0))**2).sum()
    for l in layers:
        A,b=probes[l]; pred=(np.asarray(H_by_layer[l],np.float64)@A.T)+b
        ss_res=((Hf-pred)**2).sum(); out[l]=float(1-ss_res/(ss_tot+1e-12))
    return out

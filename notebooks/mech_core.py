"""
mech_core.py — primitives for the mechanistic and mitigation experiments built on read/write.
Pure numpy; the model-side extraction (hidden states, component outputs, generation) lives in the
notebooks. Honest-scope notes are inline where a method is an approximation or an automatic proxy.
"""
import numpy as np

def unit(v):
    v=np.asarray(v,np.float64); n=np.linalg.norm(v); return v/n if n>0 else v
def proj(h,u):
    return float(np.dot(np.asarray(h,np.float64), unit(u)))

# ---------- (exp 1) answer/competitor decomposition at the final layer ----------
def logit_decomp(h_final, Wu, token_id):
    """z_token = projection * unembed_norm, with h_final already the post-final-norm residual."""
    w=np.asarray(Wu[token_id],np.float64); nu=float(np.linalg.norm(w))
    pi=float(np.dot(np.asarray(h_final,np.float64), w/nu)) if nu>0 else 0.0
    return dict(projection=pi, unembed_norm=nu, logit=pi*nu)

def final_margin(z, answer_id):
    """Answer logit minus the top competing logit (>0 means the answer already wins)."""
    z=np.asarray(z,np.float64); za=z[answer_id]
    zc=z.copy(); zc[answer_id]=-np.inf
    return float(za - zc.max())

# ---------- (exp 2) hard type-decoy margin ----------
def priming_margin_mean(read_answer, read_decoys):
    d=np.asarray(read_decoys,np.float64)
    return float(read_answer-np.mean(d)) if d.size else float("nan")
def priming_margin_hard(read_answer, read_decoys):
    """Stricter control: answer read minus the BEST decoy read (max, not mean)."""
    d=np.asarray(read_decoys,np.float64)
    return float(read_answer-np.max(d)) if d.size else float("nan")

# ---------- (exp 3) layerwise trajectory of specified tokens ----------
def layerwise_trace(z_layers, token_ids):
    """z_layers: (n_layers, V) logit-lens logits per layer. Returns per-token arrays of
    logit, rank, and log-prob across layers."""
    if z_layers is None or len(z_layers)==0:
        return {int(t):{"logit":np.array([]),"rank":np.array([]),"logprob":np.array([])} for t in token_ids}
    res={int(t):{"logit":[],"rank":[],"logprob":[]} for t in token_ids}
    for z in z_layers:
        z=np.asarray(z,np.float64); m=z.max(); lse=m+np.log(np.exp(z-m).sum())
        for t in token_ids:
            ti=int(t)
            res[ti]["logit"].append(float(z[ti]))
            res[ti]["rank"].append(int((z>z[ti]).sum()))
            res[ti]["logprob"].append(float(z[ti]-lse))
    return {t:{k:np.asarray(v) for k,v in d.items()} for t,d in res.items()}

def classify_trajectory(ans_rank, comp_rank):
    """Categorize how a failure unfolds across layers from the answer/competitor rank traces."""
    a=np.asarray(ans_rank); c=np.asarray(comp_rank); L=len(a)
    if L<3: return "short"
    mid=slice(max(1,int(0.4*L)), max(2,int(0.9*L)))
    ans_best_mid=a[mid].min() if a[mid].size else a.min()
    if ans_best_mid<=5 and a[-1]>ans_best_mid+5:
        return "answer_overwritten"            # answer surfaces midstream then gets overwritten
    if (c<a).mean()>0.7:
        return "competitor_dominant"           # competitor below answer almost throughout
    cross=np.where(c<a)[0]
    if cross.size and cross[0]>0.6*L:
        return "competitor_late_cross"         # both improve; competitor overtakes only late
    return "other"

# ---------- (exp 4) direct logit attribution of components ----------
def direct_logit_attribution(component_outputs, Wu, ln_scale, token_id):
    """Approximate DLA: contribution of each residual-space component output to a token's logit,
    component_outputs: {name: vector} at the last position (attn/mlp per layer).
    ln_scale: per-dim vector approximating the final-norm linearization (norm weight / rms of the
    full final residual). NOTE: RMSNorm is not exactly linear, so this is an approximation; use it
    for relative attribution across components, not exact logit reconstruction."""
    w=np.asarray(Wu[token_id],np.float64); s=np.asarray(ln_scale,np.float64)
    return {name: float(np.dot(s*np.asarray(v,np.float64), w)) for name,v in component_outputs.items()}

# ---------- (exp 5) patch variants (h is the post-final-norm residual) ----------
def patch_set_projection(h,u,m):
    u=unit(u); h=np.asarray(h,np.float64); return h-np.dot(h,u)*u+m*u
def patch_subtract(h,u,alpha):
    u=unit(u); h=np.asarray(h,np.float64); return h-alpha*u
def redecode_argmax(h_patched, Wu, content_mask=None):
    z=np.asarray(Wu,np.float64)@np.asarray(h_patched,np.float64)
    if content_mask is not None: z=np.where(np.asarray(content_mask,bool),z,-np.inf)
    return int(np.argmax(z))
def answer_rank(h_patched, Wu, answer_id):
    z=np.asarray(Wu,np.float64)@np.asarray(h_patched,np.float64)
    return int((z>z[answer_id]).sum())

# ---------- (exp 9) oracle top-k recall ----------
def in_topk(cand_ids, scores, gold_id, k):
    cand_ids=list(cand_ids); order=np.argsort(-np.asarray(scores,np.float64))
    return int(gold_id in [cand_ids[i] for i in order[:k]])

# ---------- (exp 10) reranker feature row for a (item, candidate) pair ----------
def candidate_features(cand_id, final_lp, read_int, freq, type_ids, prime_margin_fn, ans_len):
    """Feature vector for a candidate token: final log-prob, intermediate read, type-pool membership,
    frequency, type-read margin, answer length. prime_margin_fn(cand_id)->float."""
    return np.array([
        float(final_lp[cand_id]),
        float(read_int[cand_id]),
        float(cand_id in set(type_ids)),
        float(freq[cand_id]),
        float(prime_margin_fn(cand_id)),
        float(ans_len),
    ],dtype=np.float64)

# ---------- shared: AUROC (also in decode_core; duplicated for standalone use) ----------
def auroc(scores, labels):
    s=np.asarray(scores,np.float64); y=np.asarray(labels,int)
    n1=int(y.sum()); n0=int((1-y).sum())
    if n1==0 or n0==0: return float("nan")
    order=np.argsort(s); ranks=np.empty(len(s),float); i=0
    while i<len(s):
        j=i
        while j+1<len(s) and s[order[j+1]]==s[order[i]]: j+=1
        ranks[order[i:j+1]]=(i+j)/2.0+1.0; i=j+1
    return float((ranks[y==1].sum()-n1*(n1+1)/2.0)/(n1*n0))


def spearman(x,y):
    """Spearman rank correlation (ties averaged)."""
    x=np.asarray(x,np.float64); y=np.asarray(y,np.float64)
    if len(x)<3: return float("nan")
    def rank(a):
        o=np.argsort(a); r=np.empty(len(a),float); i=0
        while i<len(a):
            j=i
            while j+1<len(a) and a[o[j+1]]==a[o[i]]: j+=1
            r[o[i:j+1]]=(i+j)/2.0+1.0; i=j+1
        return r
    rx,ry=rank(x),rank(y)
    rx-=rx.mean(); ry-=ry.mean()
    d=np.sqrt((rx**2).sum()*(ry**2).sum())
    return float((rx*ry).sum()/d) if d>0 else float("nan")


# ---------- single-token / prefix robustness primitives ----------
def is_single_token(answer_text, tokenizer_encode):
    """True if the answer is a single token under the leading-space convention."""
    ids=tokenizer_encode(" "+answer_text)
    return len(ids)==1

def first_token_unique(gold_id, alias_first_ids, decoy_first_ids):
    """True if the gold's first token uniquely identifies the answer: it must NOT collide with any
    SAME-RELATION DECOY's first token (a real prefix ambiguity), while sharing with the gold's own
    alias variants is allowed (those are correct). Returns False (prefix-ambiguous) if a decoy shares
    the gold's first token."""
    decoys=set(int(x) for x in decoy_first_ids if x is not None)
    return int(gold_id) not in decoys


# ---------- bootstrap CI ----------
def bootstrap_ci(values, stat=np.mean, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for a statistic over a 1-D sample."""
    v=np.asarray([x for x in values if x is not None and np.isfinite(x)],float)
    if len(v)<3: return dict(point=float(stat(v)) if len(v) else float("nan"), lo=float("nan"), hi=float("nan"), n=int(len(v)))
    rng=np.random.default_rng(seed)
    bs=np.array([stat(v[rng.integers(0,len(v),len(v))]) for _ in range(n_boot)])
    return dict(point=float(stat(v)), lo=float(np.percentile(bs,100*alpha/2)),
                hi=float(np.percentile(bs,100*(1-alpha/2))), n=int(len(v)))

def bootstrap_ci_spearman(x, y, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for Spearman rho (resampling paired observations)."""
    x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
    if len(x)<5: return dict(point=spearman(x,y), lo=float("nan"), hi=float("nan"), n=int(len(x)))
    rng=np.random.default_rng(seed); bs=[]
    for _ in range(n_boot):
        idx=rng.integers(0,len(x),len(x)); bs.append(spearman(x[idx],y[idx]))
    bs=np.array([b for b in bs if np.isfinite(b)])
    return dict(point=spearman(x,y), lo=float(np.percentile(bs,100*alpha/2)),
                hi=float(np.percentile(bs,100*(1-alpha/2))), n=int(len(x)))

# ---------- multi-token / prefix primitives ----------
def stricter_prefix_unique(gold_first_ids, gold_first2_ids, decoy_first_ids, decoy_first2_ids,
                           gold_id, min_char_len=3, decode=None):
    """Return a dict of several prefix-uniqueness criteria for one item:
    - unique1_decoy: gold first token not shared by any same-relation decoy first token
    - unique1_alias: gold first token not shared across the gold's OWN alias variants beyond itself
      (a weak alias may share; informational only)
    - unique2_decoy: gold first TWO tokens not a prefix of any decoy's first two tokens
    - long_enough: gold first token decodes to >= min_char_len non-space chars
    """
    decoys=set(int(x) for x in decoy_first_ids if x is not None)
    u1=int(gold_id) not in decoys
    dec2=set(tuple(p) for p in decoy_first2_ids if p)
    g2=tuple(gold_first2_ids) if gold_first2_ids else None
    u2=(g2 is not None) and (g2 not in dec2)
    long_enough=True
    if decode is not None:
        s=decode([int(gold_id)]).strip()
        long_enough=len(s)>=min_char_len
    return dict(unique1_decoy=bool(u1), unique2_decoy=bool(u2), long_enough=bool(long_enough),
                strict=bool(u1 and u2 and long_enough))

def sequence_logprob(per_token_logprobs):
    """Sum and mean log-prob of a gold answer token sequence (teacher-forced)."""
    a=np.asarray(per_token_logprobs,float)
    return dict(sum=float(a.sum()), mean=float(a.mean()) if a.size else float("nan"), n_tokens=int(a.size))

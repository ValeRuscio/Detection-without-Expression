"""
rw_core.py — the read/write dissociation, WITHOUT any 'recognition' construct.

No knowledge label. Two direct, mechanical quantities measured on ONE forward pass of the question:
  READ  = how decodable the answer is from the residual stream (logit-lens), measured two ways:
          - intermediate: PEAK decodability across mid-stream layers (NON-circular: early presence)
          - final: decodability at the last layer (near-trivial: ~ projection*norm = the output)
  WRITE = the answer's standing in the actual OUTPUT (final logit rank / does it win).

Phenomenon = answer READABLE mid-stream but does NOT WIN the output. The intermediate-vs-final
contrast shows whether the dissociation is real (depth-separated) or trivial (lives in the norm).

Priming control: 'readable' must be answer-SPECIFIC, not answer-TYPE priming (the Volvo problem).
Compare the correct answer's mid-stream decodability to TYPE-MATCHED decoys; the read signal is
meaningful only if the correct answer beats type-matched decoys mid-stream.

Self-contained. Tested in test_rw_core.py.
"""
import numpy as np
EPS = 1e-9

def log_softmax_1d(z):
    z = np.asarray(z, np.float64); m = z.max(); return z - (m + np.log(np.exp(z-m).sum()+EPS))

def decodability_at_layer(hidden_vec, W_U, token_ids):
    """logit-lens logprob + rank of the answer token at one layer's residual."""
    z = np.asarray(W_U, np.float64) @ np.asarray(hidden_vec, np.float64)
    lp = log_softmax_1d(z)
    order = np.argsort(-z); rank = np.empty_like(order); rank[order] = np.arange(len(order))
    best_lp = max(float(lp[t]) for t in token_ids)
    best_rank = min(int(rank[t]) for t in token_ids)
    return dict(logprob=best_lp, rank=best_rank)

def read_write_record(hidden_by_layer, W_U, answer_ids, intermediate_frac=(0.4, 0.9)):
    """hidden_by_layer: list of per-layer last-token residual vectors (len L+1, index 0 = embeddings).
    Returns:
      read_intermediate = PEAK answer-logprob across the intermediate band (frac of depth)
      read_final        = answer-logprob at the final layer (pre-unembedding residual)
      write_rank        = answer's rank in the final OUTPUT (lower = wins)
      write_is_top      = answer is the argmax of the final output
    """
    L = len(hidden_by_layer)
    lo = int(intermediate_frac[0]*L); hi = int(intermediate_frac[1]*L)
    inter_lps = []
    for li in range(max(1, lo), max(lo+1, hi)):
        inter_lps.append(decodability_at_layer(hidden_by_layer[li], W_U, answer_ids)["logprob"])
    read_inter = float(np.max(inter_lps)) if inter_lps else np.nan
    fin = decodability_at_layer(hidden_by_layer[-1], W_U, answer_ids)
    return dict(read_intermediate=read_inter, read_final=fin["logprob"],
                write_rank=fin["rank"], write_is_top=int(fin["rank"] == 0))

def priming_control(hidden_by_layer, W_U, answer_ids, decoy_ids_list, intermediate_frac=(0.4, 0.9)):
    """Is the answer's mid-stream decodability ANSWER-SPECIFIC or just type priming? Compare the
    answer's peak intermediate logprob to the peak intermediate logprob of TYPE-MATCHED decoys.
    Positive margin => the read signal carries fact-specific info, not just answer-type priming."""
    L = len(hidden_by_layer); lo = int(intermediate_frac[0]*L); hi = int(intermediate_frac[1]*L)
    def peak(ids):
        v = [decodability_at_layer(hidden_by_layer[li], W_U, ids)["logprob"] for li in range(max(1,lo), max(lo+1,hi))]
        return float(np.max(v)) if v else np.nan
    ans = peak(answer_ids)
    decoys = [peak([d]) for d in decoy_ids_list]
    decoys = [x for x in decoys if np.isfinite(x)]
    if not decoys or not np.isfinite(ans): return np.nan
    return float(ans - np.mean(decoys))

# ---------- dissociation summary ----------
def dissociation_rate(records, read_key, read_threshold, winning=False):
    """Fraction of items that are READABLE (read_key logprob above threshold) but do NOT WIN the
    output (write_is_top==0). The core phenomenon. With winning=True, returns readable-AND-winning."""
    rec = [r for r in records if np.isfinite(r.get(read_key, np.nan))]
    if not rec: return dict(rate=np.nan, n=0)
    readable = [r for r in rec if r[read_key] >= read_threshold]
    if not readable: return dict(rate=np.nan, n=0, n_readable=0)
    hits = sum((r["write_is_top"] == (1 if winning else 0)) for r in readable)
    return dict(rate=float(hits/len(readable)), n=len(readable), n_readable=len(readable))

def read_write_correlation(records, read_key):
    """Across items: does READ predict WRITE? If read_intermediate barely predicts write_rank, the
    dissociation is strong (readable items often still lose). For read_final it should predict write
    strongly (near-trivial). Returns Spearman-like rank correlation of read vs (negative) write_rank."""
    from scipy.stats import spearmanr
    rd = np.array([r[read_key] for r in records if np.isfinite(r.get(read_key,np.nan)) and np.isfinite(r.get("write_rank",np.nan))], float)
    wr = np.array([r["write_rank"] for r in records if np.isfinite(r.get(read_key,np.nan)) and np.isfinite(r.get("write_rank",np.nan))], float)
    if len(rd) < 5: return np.nan
    rho,_ = spearmanr(rd, -wr)   # higher read vs better (lower) rank
    return float(rho)

def class_means(values, classes, names=("retrieved","gap","absent")):
    values=np.asarray(values,float); classes=np.asarray(classes); out={}
    for c,nm in enumerate(names):
        v=values[classes==c]; v=v[np.isfinite(v)]
        out[nm]=(float(np.mean(v)) if len(v) else np.nan, int(len(v)))
    return out


# ---------- 2x2 joint competitor breakdown (frequency route x type-confusion route) ----------
def competitor_2x2(items_winner_info):
    """Disentangle the two overlapping competitor routes. Input: list of dicts per readable-losing
    item, each with:
       higher_freq : bool  (winner higher corpus-frequency than the answer)
       type_matched: bool  (winner is a same-relation / same-type token)
    Returns the 2x2 joint distribution + marginals, so 'type-confusion independent of frequency'
    can be read off directly instead of from two overlapping percentages."""
    hf = np.array([bool(d["higher_freq"]) for d in items_winner_info])
    tm = np.array([bool(d["type_matched"]) for d in items_winner_info])
    n = len(hf)
    if n == 0: return dict(n=0)
    cell = lambda a, b: int(np.sum((hf == a) & (tm == b)))
    out = dict(
        n=n,
        both          = cell(True,  True),    # higher-freq AND same-type
        freq_only     = cell(True,  False),   # higher-freq, NOT same-type (generic frequency default)
        type_only     = cell(False, True),    # same-type, NOT higher-freq (pure type-confusion)
        neither       = cell(False, False),   # neither route
        marg_higher_freq  = float(hf.mean()),
        marg_type_matched = float(tm.mean()),
    )
    out["frac_type_indep_of_freq"] = out["type_only"]/n     # type-confusion that frequency can't explain
    out["frac_freq_indep_of_type"] = out["freq_only"]/n     # frequency that type-confusion can't explain
    return out

def fmt_2x2(d):
    if d.get("n",0)==0: return "(empty)"
    n=d["n"]
    return (f"n={n}\n"
            f"                      type-matched   NOT type-matched\n"
            f"  higher-freq        {d['both']:>8} ({d['both']/n:.2f})   {d['freq_only']:>8} ({d['freq_only']/n:.2f})\n"
            f"  NOT higher-freq    {d['type_only']:>8} ({d['type_only']/n:.2f})   {d['neither']:>8} ({d['neither']/n:.2f})\n"
            f"  pure type-confusion (type, not freq): {d['frac_type_indep_of_freq']:.3f}\n"
            f"  pure frequency-default (freq, not type): {d['frac_freq_indep_of_type']:.3f}")


# ---------- content-token winner (exclude format tokens: newlines, punct, whitespace, EOS) ----------
def content_winner(logits, content_mask):
    """The competitor should be the top CONTENT token, not the raw argmax (which is often a format
    token like a newline). content_mask: bool array over vocab, True where the token's decoded form
    contains an alphanumeric character (and is not special/byte). Returns the raw argmax, whether it
    was a format token (the stop-vs-confabulate signal), and the top content token."""
    z = np.asarray(logits, np.float64); cm = np.asarray(content_mask, bool)
    raw = int(np.argmax(z))
    raw_is_format = (not bool(cm[raw]))
    zc = np.where(cm, z, -np.inf)
    cw = int(np.argmax(zc)) if np.isfinite(zc).any() else raw
    return dict(raw=raw, raw_is_format=raw_is_format, content_winner=cw)

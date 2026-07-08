"""
olmo_runner.py — OLMo training-checkpoint trajectory for the frequency-geometry claim. For each
checkpoint: norm-frequency correlation, sign-stable directional-frequency correlation, baseline-logit
frequency correlation, and spectral anisotropy (effective rank after projecting out the frequency
direction). The two-effect phrasing: co-evolution builds directionality (established early, ~flat);
weight decay suppresses norm-frequency coupling (decays over training).
"""
import numpy as np, re, gc

def _erank(S):
    p=(S**2); s=p.sum()
    if s<=0: return float("nan")
    p=p/s; H=-(p*np.log(p+1e-12)).sum(); return float(np.exp(H))

def measure_checkpoint(Wu, freq, ref_dir):
    """Wu: (V,d) unembedding (float64). freq: (V,) log-freq. ref_dir: fixed reference frequency
    direction (unit) for the sign-stable directional coupling. Returns the four trajectory measures."""
    Wu=np.asarray(Wu,np.float64); f=np.asarray(freq,np.float64)
    norms=np.linalg.norm(Wu,axis=1)
    norm_freq=float(np.corrcoef(norms,f)[0,1]) if norms.std()>0 else float("nan")
    proj=Wu@ref_dir
    dir_freq=float(np.corrcoef(proj,f)[0,1]) if proj.std()>0 else float("nan")
    # baseline-logit frequency: logit of each token from a zero/neutral hidden = bias-like; use mean
    # unembedding projection of the mean row direction as a proxy for the readout's frequency tilt
    mean_dir=Wu.mean(0); base_logit=Wu@ (mean_dir/ (np.linalg.norm(mean_dir)+1e-12))
    base_freq=float(np.corrcoef(base_logit,f)[0,1]) if base_logit.std()>0 else float("nan")
    # spectral anisotropy AFTER projecting out the frequency direction
    Wp=Wu-(Wu@ref_dir)[:,None]*ref_dir[None,:]
    Wc=Wp-Wp.mean(0,keepdims=True)
    S=np.linalg.svd(Wc,compute_uv=False)
    er=_erank(S); spec=1.0-er/Wu.shape[1]
    return dict(norm_freq=norm_freq, dir_freq=dir_freq, base_freq=base_freq,
                effective_rank=er, spectral_anisotropy=spec)

def reference_direction(Wu_final, freq):
    f=np.asarray(freq,np.float64); fbar=f.mean()
    r=((f-fbar)[:,None]*np.asarray(Wu_final,np.float64)).sum(0)
    n=np.linalg.norm(r); return r/n if n>0 else r

def discover_checkpoints(model_id="allenai/OLMo-1B-hf", want=None):
    """List available training-step revisions (e.g. step1000-tokens...) sorted by step."""
    from huggingface_hub import list_repo_refs
    refs=list_repo_refs(model_id)
    steps=[]
    for b in refs.branches:
        mobj=re.search(r"step(\d+)", b.name)
        if mobj: steps.append((int(mobj.group(1)), b.name))
    steps.sort()
    if want:  # nearest available to each requested step
        chosen=[]
        for w in want:
            chosen.append(min(steps,key=lambda s:abs(s[0]-w)))
        seen=set(); out=[]
        for s in chosen:
            if s[0] not in seen: out.append(s); seen.add(s[0])
        return out
    return steps

def run_olmo_trajectory(model_id="allenai/OLMo-1B-hf", device="cuda", dtype=None,
                        want_steps=(500,1000,2000,5000,10000,20000,50000,100000,200000,400000,738000),
                        freq=None, tokenizer=None, freq_lines=8000):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from collections import Counter
    dtype=dtype or (torch.float16 if device=="cuda" else torch.float32)
    tk=tokenizer or AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
    # frequency table (once, from final tokenizer)
    if freq is None:
        counts=Counter(); n=0
        for r in load_dataset("wikitext","wikitext-103-raw-v1",split="train",streaming=True):
            t=r["text"].strip()
            if len(t)>40: counts.update(tk(t,add_special_tokens=False,truncation=True,max_length=256)["input_ids"]); n+=1
            if n>=freq_lines: break
        # size set after first model load; placeholder, filled below
    ckpts=discover_checkpoints(model_id, want=want_steps)
    # load final checkpoint first to build the fixed reference direction and freq vector size
    rows=[]; ref_dir=None; freq_vec=None
    # final = last step
    all_steps=[s for s,_ in ckpts]
    for step,rev in ckpts:
        m=AutoModelForCausalLM.from_pretrained(model_id,revision=rev,torch_dtype=dtype,
                                               device_map=device,trust_remote_code=True).eval()
        for p in m.parameters(): p.requires_grad_(False)
        Wu=m.get_output_embeddings().weight.detach().float().cpu().numpy().astype(np.float64)
        V=Wu.shape[0]
        if freq_vec is None:
            from collections import Counter as C
            counts=C(); n=0
            for r in load_dataset("wikitext","wikitext-103-raw-v1",split="train",streaming=True):
                t=r["text"].strip()
                if len(t)>40: counts.update(tk(t,add_special_tokens=False,truncation=True,max_length=256)["input_ids"]); n+=1
                if n>=freq_lines: break
            tot=sum(counts.values()); freq_vec=np.full(V,np.log(1/(tot+V)))
            for k,v in counts.items():
                if k<V: freq_vec[k]=np.log((v+1)/(tot+V))
        if ref_dir is None and step==max(all_steps):
            ref_dir=reference_direction(Wu,freq_vec)
        del m; torch.cuda.empty_cache(); gc.collect()
    # build ref_dir from the final checkpoint explicitly (second pass guards step ordering)
    if ref_dir is None:
        step,rev=ckpts[-1]
        m=AutoModelForCausalLM.from_pretrained(model_id,revision=rev,torch_dtype=dtype,device_map=device,trust_remote_code=True).eval()
        Wu=m.get_output_embeddings().weight.detach().float().cpu().numpy().astype(np.float64)
        ref_dir=reference_direction(Wu,freq_vec); del m; torch.cuda.empty_cache(); gc.collect()
    # measure each checkpoint
    for step,rev in ckpts:
        m=AutoModelForCausalLM.from_pretrained(model_id,revision=rev,torch_dtype=dtype,device_map=device,trust_remote_code=True).eval()
        Wu=m.get_output_embeddings().weight.detach().float().cpu().numpy().astype(np.float64)
        meas=measure_checkpoint(Wu,freq_vec,ref_dir); meas["step"]=step; rows.append(meas)
        del m; torch.cuda.empty_cache(); gc.collect()
        print(f"  step {step}: norm_freq={meas['norm_freq']:+.3f} dir_freq={meas['dir_freq']:+.3f} "
              f"base_freq={meas['base_freq']:+.3f} erank={meas['effective_rank']:.0f}",flush=True)
    rows.sort(key=lambda r:r["step"])
    return rows

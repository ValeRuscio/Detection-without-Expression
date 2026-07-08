"""
mech_runner.py — per-model runners for the read/write mechanism experiments, so the cross-model
notebook can loop over models cleanly. Each experiment takes a `ctx` (built by make_ctx) and returns
a compact summary dict. Includes exp4b: causal late-attention ablation (mean-ablation), which upgrades
the exp4 attribution from correlational to causal.

Heavy parts (model, generation, hooks) live here; pure-numpy scoring is delegated to mech_core/rw_core.
"""
import numpy as np, re, gc, torch
import rw_core as rw, mech_core as mc

# ---------------- context construction ----------------
def make_ctx(name, device, dtype, items, rel, qa, templates, freq_lines=8000, alnum=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from collections import Counter
    alnum=alnum or re.compile(r"[A-Za-z0-9]")
    tk=AutoTokenizer.from_pretrained(name,trust_remote_code=True); tk.padding_side="left"
    if tk.pad_token is None: tk.pad_token=tk.eos_token
    kw=dict(torch_dtype=dtype,output_hidden_states=True,device_map=device,trust_remote_code=True)
    try: m=AutoModelForCausalLM.from_pretrained(name,**kw).eval()
    except Exception:
        kw["torch_dtype"]=torch.bfloat16; m=AutoModelForCausalLM.from_pretrained(name,**kw).eval()
    Wu_t=m.get_output_embeddings().weight.detach()
    Wu_np=Wu_t.detach().float().cpu().numpy().astype(np.float64)
    fn=m.model.norm if hasattr(m.model,"norm") else m.model.final_layernorm
    V=Wu_np.shape[0]; L=m.config.num_hidden_layers
    read_band=[l for l in range(L+1) if 0.4<=l/L<=0.9]
    late_band=[l for l in range(1,L) if 0.80<=l/L<1.0]      # late decoder layers (model-relative)
    mid_band =[l for l in range(1,L) if 0.40<=l/L<=0.60]    # control band
    # content mask
    mask=np.zeros(V,bool); sp=set(tk.all_special_ids or []); Vt=min(V,tk.vocab_size if hasattr(tk,"vocab_size") else len(tk))
    for s in range(0,Vt,4000):
        ids=list(range(s,min(s+4000,Vt)))
        try:
            for i,dd in zip(ids,tk.batch_decode([[i] for i in ids])):
                if i not in sp: mask[i]=bool(alnum.search(dd))
        except Exception:
            for i in ids:
                if i not in sp:
                    try: mask[i]=bool(alnum.search(tk.decode([i])))
                    except Exception: mask[i]=False
    # frequency
    counts=Counter(); n=0
    for r in load_dataset("wikitext","wikitext-103-raw-v1",split="train",streaming=True):
        t=r["text"].strip()
        if len(t)>40: counts.update(tk(t,add_special_tokens=False,truncation=True,max_length=256)["input_ids"]); n+=1
        if n>=freq_lines: break
    tot=sum(counts.values()); freq=np.full(V,np.log(1/(tot+V)))
    for k,v in counts.items():
        if k<V: freq[k]=np.log((v+1)/(tot+V))
    _fid={}
    def fid(w):
        if w in _fid: return _fid[w]
        t=tk(" "+w,add_special_tokens=False)["input_ids"]; r=t[0] if t else None; _fid[w]=r; return r
    relpool={rl:sorted({i for i in (fid(a) for a in set(ans)) if i is not None}) for rl,ans in rel.items()}
    n_heads=getattr(m.config,'num_attention_heads',None)
    head_dim=(m.config.hidden_size//n_heads) if n_heads else None
    return dict(name=name,m=m,tk=tk,Wu_np=Wu_np,Wu_t=Wu_t,fn=fn,V=V,L=L,device=device,
                n_heads=n_heads,head_dim=head_dim,hidden=m.config.hidden_size,
                read_band=read_band,late_band=late_band,mid_band=mid_band,content_mask=mask,
                freq=freq,fid=fid,relpool=relpool,qa=qa,templates=templates,items=items)

def free_ctx(ctx):
    try: del ctx["m"], ctx["Wu_t"]
    except Exception: pass
    torch.cuda.empty_cache(); gc.collect()

# ---------------- shared text helpers ----------------
def _na(s):
    s=s.lower().strip(); s=re.sub(r"\b(a|an|the)\b"," ",s); s=re.sub(r"[^a-z0-9 ]"," ",s); return re.sub(r"\s+"," ",s).strip()
def _match(text,gold): return any(_na(x) in _na(text) or _na(text) in _na(x) for x in gold)

@torch.no_grad()
def _greedy(ctx,prompt,maxnew=12):
    tk=ctx["tk"]; enc=tk(prompt,return_tensors="pt",truncation=True,max_length=256).to(ctx["device"])
    g=ctx["m"].generate(**enc,max_new_tokens=maxnew,do_sample=False,num_beams=1,pad_token_id=tk.pad_token_id)
    return tk.decode(g[0,enc["input_ids"].shape[1]:],skip_special_tokens=True)

def _lens_logprob(ctx,hidden_last_prenorm):
    h=ctx["fn"](hidden_last_prenorm).float()
    z=(h@ctx["Wu_t"].float().T)
    return torch.log_softmax(z,dim=-1).detach().cpu().numpy()

@torch.no_grad()
def _run_prompt(ctx,prompt,want_alllayers=False):
    tk=ctx["tk"]; enc=tk(prompt,return_tensors="pt",truncation=True,max_length=256).to(ctx["device"])
    out=ctx["m"](**enc); hs=out.hidden_states; L=ctx["L"]
    h_final=ctx["fn"](hs[L][0,-1,:]).float().detach().cpu().numpy()
    z_final=out.logits[0,-1,:].float().detach().cpu().numpy()
    band=list(range(L+1)) if want_alllayers else ctx["read_band"]
    r_int=np.max(np.stack([_lens_logprob(ctx,hs[l][0,-1,:]) for l in ctx["read_band"]]),axis=0)
    z_layers=[ (ctx["fn"](hs[l][0,-1,:]).float()@ctx["Wu_t"].float().T).detach().cpu().numpy() for l in band] if want_alllayers else None
    return dict(h_final=h_final,z_final=z_final,r_int=r_int,z_layers=z_layers)

# ---------------- experiment 1: paired paraphrase ----------------
def exp1_paired(ctx,max_facts=80):
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; T=ctx["templates"]
    pairs=[]
    for it in items:
        aid=fid(it["gold"][0])
        if aid is None: continue
        succ=fail=None
        for tpl in T:
            p=tpl.format(q=it["q"]); ok=_match(_greedy(ctx,p),it["gold"])
            if ok and succ is None: succ=p
            if (not ok) and fail is None: fail=p
            if succ and fail: break
        if succ and fail: pairs.append((it,aid,succ,fail))
        if len(pairs)>=max_facts: break
    rows=[]; recov=0
    for it,aid,succ,fail in pairs:
        S=_run_prompt(ctx,succ); F=_run_prompt(ctx,fail)
        cwF=rw.content_winner(F["z_final"],ctx["content_mask"])["content_winner"]
        dS=mc.logit_decomp(S["h_final"],ctx["Wu_np"],aid); dF=mc.logit_decomp(F["h_final"],ctx["Wu_np"],aid)
        rows.append(dict(read_succ=float(S["r_int"][aid]),read_fail=float(F["r_int"][aid]),
                         proj_ans_succ=dS["projection"],proj_ans_fail=dF["projection"],
                         proj_comp_fail=mc.logit_decomp(F["h_final"],ctx["Wu_np"],cwF)["projection"],
                         margin_succ=mc.final_margin(S["z_final"],aid),margin_fail=mc.final_margin(F["z_final"],aid)))
        hp=mc.patch_set_projection(F["h_final"],mc.unit(ctx["Wu_np"][aid]),dS["projection"])
        recov+=int(mc.redecode_argmax(hp,ctx["Wu_np"],ctx["content_mask"])==aid)
    if not rows: return dict(n_pairs=0)
    M={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
    M["n_pairs"]=len(pairs); M["crosspatch_recovery"]=recov/max(len(pairs),1)
    return M

# ---------------- experiment 2: hard decoy ----------------
def exp2_hard_decoy(ctx,max_items=300):
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]
    mean_pos=hard_pos=nfail=0
    for it in items:
        aid=fid(it["gold"][0])
        if aid is None: continue
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]): continue
        R=_run_prompt(ctx,p); ra=float(R["r_int"][aid])
        pool=[i for i in ctx["relpool"][it["rel"]] if i!=aid]
        rng=np.random.default_rng(7); dk=list(rng.choice(pool,min(8,len(pool)),replace=False)) if pool else []
        if not dk: continue
        rd=[float(R["r_int"][d]) for d in dk]; nfail+=1
        mean_pos+=int(mc.priming_margin_mean(ra,rd)>0); hard_pos+=int(mc.priming_margin_hard(ra,rd)>0)
        if nfail>=max_items: break
    return dict(n=nfail,readable_mean=mean_pos/max(nfail,1),readable_hard=hard_pos/max(nfail,1))

# ---------------- experiment 3: trajectory ----------------
def exp3_trajectory(ctx,max_items=120):
    from collections import Counter
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; cls=Counter(); n=0
    for it in items:
        aid=fid(it["gold"][0])
        if aid is None: continue
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]): continue
        R=_run_prompt(ctx,p,want_alllayers=True)
        cw=rw.content_winner(R["z_final"],ctx["content_mask"])["content_winner"]
        if cw==aid: continue
        tr=mc.layerwise_trace(R["z_layers"],[aid,cw])
        cls[mc.classify_trajectory(tr[aid]["rank"],tr[cw]["rank"])]+=1; n+=1
        if n>=max_items: break
    return dict(n=n,**{k:cls.get(k,0) for k in ["competitor_dominant","answer_overwritten","competitor_late_cross","other","short"]})

# ---------------- experiment 4: attribution (DLA) ----------------
def _capture(ctx,prompt,attn_layers,mlp_layers):
    tk=ctx["tk"]; m=ctx["m"]; store={}; handles=[]
    def mk(name,kind):
        def hook(mod,inp,out):
            o=out[0] if isinstance(out,tuple) else out
            store[name]=o[0,-1,:].detach().float().detach().cpu().numpy()
        return hook
    for li in attn_layers: handles.append(m.model.layers[li].self_attn.register_forward_hook(mk(f"attn.{li}","attn")))
    for li in mlp_layers:  handles.append(m.model.layers[li].mlp.register_forward_hook(mk(f"mlp.{li}","mlp")))
    enc=tk(prompt,return_tensors="pt",truncation=True,max_length=256).to(ctx["device"])
    with torch.no_grad(): out=m(**enc)
    for h in handles: h.remove()
    hs=out.hidden_states; L=ctx["L"]
    h_pre=hs[L][0,-1,:]
    w_ln=getattr(ctx["fn"],"weight",None)
    rms=float(torch.sqrt((h_pre.float()**2).mean()+1e-6).cpu())
    ln_scale=(w_ln.detach().float().detach().cpu().numpy()/rms) if w_ln is not None else np.ones(ctx["Wu_np"].shape[1])/rms
    return dict(z_final=out.logits[0,-1,:].float().detach().cpu().numpy(),components=store,ln_scale=ln_scale)

def exp4_attribution(ctx,max_items=100):
    from collections import defaultdict
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; L=ctx["L"]
    layers=ctx["late_band"]+ctx["mid_band"]
    agg=defaultdict(lambda:[0.0,0.0]); n=0
    for it in items:
        aid=fid(it["gold"][0])
        if aid is None: continue
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]): continue
        cap=_capture(ctx,p,layers,layers)
        cw=rw.content_winner(cap["z_final"],ctx["content_mask"])["content_winner"]
        if cw==aid: continue
        a=mc.direct_logit_attribution(cap["components"],ctx["Wu_np"],cap["ln_scale"],aid)
        c=mc.direct_logit_attribution(cap["components"],ctx["Wu_np"],cap["ln_scale"],cw)
        for nm in a: agg[nm][0]+=a[nm]; agg[nm][1]+=c[nm]
        n+=1
        if n>=max_items: break
    rows=[dict(component=nm,attr_answer=s[0]/max(n,1),attr_competitor=s[1]/max(n,1),comp_minus_ans=(s[1]-s[0])/max(n,1)) for nm,s in agg.items()]
    rows.sort(key=lambda r:-r["comp_minus_ans"])
    return dict(n=n,rows=rows[:12])

# ---------------- experiment 4b: CAUSAL late-attention ablation (mean ablation) ----------------
def _forward_ablate(ctx,prompt,specs,means):
    """specs: list of ('attn'|'mlp', layer). means: {(kind,layer): vec}. Mean-ablate the last-position
    output of each component, then read final logits."""
    tk=ctx["tk"]; m=ctx["m"]; handles=[]
    def mk(kind,layer):
        mv=means[(kind,layer)]
        def hook(mod,inp,out):
            o=out[0] if isinstance(out,tuple) else out
            o=o.clone(); o[:, -1, :]=torch.tensor(mv,dtype=o.dtype,device=o.device)
            if isinstance(out,tuple): return (o,)+tuple(out[1:])
            return o
        return hook
    for kind,layer in specs:
        mod=m.model.layers[layer].self_attn if kind=="attn" else m.model.layers[layer].mlp
        handles.append(mod.register_forward_hook(mk(kind,layer)))
    enc=tk(prompt,return_tensors="pt",truncation=True,max_length=256).to(ctx["device"])
    with torch.no_grad(): out=m(**enc)
    for h in handles: h.remove()
    return out.logits[0,-1,:].float().detach().cpu().numpy()

def exp4b_attn_ablation(ctx,max_items=120):
    """Causal test: mean-ablate late attention (vs controls) and measure the change in the competitor's
    lead over the answer and the answer recovery rate. If late-attn ablation shrinks the competitor lead
    most and recovers most, late attention causally writes the competitor."""
    import random
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; late=ctx["late_band"]; mid=ctx["mid_band"]
    rng=np.random.default_rng(0)
    rand_layer=[int(rng.choice(late))] if late else []
    # Phase A: collect readable failures + accumulate mean component outputs at needed layers
    need_attn=sorted(set(late+mid+rand_layer)); need_mlp=sorted(set(late))
    sums={("attn",l):None for l in need_attn}; sums.update({("mlp",l):None for l in need_mlp})
    fails=[]; cnt=0
    for it in items:
        aid=fid(it["gold"][0])
        if aid is None: continue
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]): continue
        cap=_capture(ctx,p,need_attn,need_mlp)
        cw=rw.content_winner(cap["z_final"],ctx["content_mask"])["content_winner"]
        if cw==aid: continue
        za=float(cap["z_final"][aid]); zc=float(cap["z_final"][cw])
        fails.append(dict(p=p,aid=aid,cw=cw,za=za,zc=zc))
        for nm,vec in cap["components"].items():
            kind,layer=nm.split("."); key=(kind,int(layer))
            if key in sums: sums[key]=vec.copy() if sums[key] is None else sums[key]+vec
        cnt+=1
        if cnt>=max_items: break
    if cnt==0: return dict(n=0)
    means={k:(v/cnt) for k,v in sums.items() if v is not None}
    # Phase B: per intervention, measure mean delta(competitor lead) and recovery
    interventions={"late_attn":[("attn",l) for l in late],
                   "mid_attn":[("attn",l) for l in mid],
                   "late_mlp":[("mlp",l) for l in late],
                   "random_attn":[("attn",l) for l in rand_layer]}
    res={}
    for name,specs in interventions.items():
        specs=[s for s in specs if s in means]
        if not specs: res[name]=dict(n=0); continue
        dlead=[]; dcomp=[]; dans=[]; recov=0
        for f in fails:
            z=_forward_ablate(ctx,f["p"],specs,means)
            za2=float(z[f["aid"]]); zc2=float(z[f["cw"]])
            dlead.append((zc2-za2)-(f["zc"]-f["za"]))   # change in competitor lead (negative = competitor lead shrinks)
            dcomp.append(zc2-f["zc"]); dans.append(za2-f["za"])
            recov+=int(np.argmax(np.where(ctx["content_mask"],z,-np.inf))==f["aid"])
        res[name]=dict(n=len(fails),d_competitor_lead=float(np.mean(dlead)),
                       d_competitor=float(np.mean(dcomp)),d_answer=float(np.mean(dans)),
                       recovery=recov/len(fails))
    return dict(n=cnt,n_late=len(late),n_mid=len(mid),interventions=res)

# ---------------- experiment 5: patching variants ----------------
def exp5_patching(ctx,max_items=200):
    from collections import defaultdict
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]
    # model-relative answer-up target from retrieved items
    rp=[]
    for it in items[:150]:
        aid=fid(it["gold"][0])
        if aid is None: continue
        p=QA.format(q=it["q"])
        if not _match(_greedy(ctx,p),it["gold"]): continue
        R=_run_prompt(ctx,p); rp.append(mc.logit_decomp(R["h_final"],ctx["Wu_np"],aid)["projection"])
    M=float(np.median(rp)) if rp else 10.0; ALPHA=M
    res=defaultdict(int); split={"low":defaultdict(int),"high":defaultdict(int)}; n=0
    for it in items:
        aid=fid(it["gold"][0])
        if aid is None: continue
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]): continue
        R=_run_prompt(ctx,p); cw=rw.content_winner(R["z_final"],ctx["content_mask"])["content_winner"]
        if cw==aid: continue
        h=R["h_final"]; ua=mc.unit(ctx["Wu_np"][aid]); uc=mc.unit(ctx["Wu_np"][cw])
        cand={"answer_up":mc.patch_set_projection(h,ua,M),
              "competitor_down":mc.patch_subtract(h,uc,ALPHA),
              "both":mc.patch_subtract(mc.patch_set_projection(h,ua,M),uc,ALPHA),
              "random":mc.patch_subtract(h,mc.unit(np.random.default_rng(n).standard_normal(h.shape[0])),ALPHA)}
        b="high" if R["r_int"][aid]>np.median(R["r_int"]) else "low"
        for nm,hp in cand.items():
            w=int(mc.redecode_argmax(hp,ctx["Wu_np"],ctx["content_mask"])==aid); res[nm]+=w; split[b][nm]+=w
        res["_n"]+=1; split[b]["_n"]+=1; n+=1
        if n>=max_items: break
    out=dict(M=M,alpha=ALPHA,n=res["_n"])
    for k in ["answer_up","competitor_down","both","random"]: out[k]=res[k]/max(res["_n"],1)
    for b in ["low","high"]:
        nb=split[b]["_n"]; out[f"{b}_n"]=nb
        out[f"{b}_competitor_down"]=split[b]["competitor_down"]/max(nb,1)
        out[f"{b}_answer_up"]=split[b]["answer_up"]/max(nb,1)
    return out


# ================= NEW EXPERIMENTS =================

# ---------------- exp 1b: frequency-controlled paired paraphrase ----------------
def exp1b_freq_controlled(ctx, freq, max_facts=80):
    """Like exp1 but report stats separately on pairs where the failure competitor is NOT more
    frequent than the answer. If projComp_fail > projA_fail still holds there, the margin effect is
    not explained by the competitor being more frequent."""
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; T=ctx["templates"]
    pairs=[]
    for it in items:
        aid=fid(it["gold"][0])
        if aid is None: continue
        succ=fail=None
        for tpl in T:
            p=tpl.format(q=it["q"]); ok=_match(_greedy(ctx,p),it["gold"])
            if ok and succ is None: succ=p
            if (not ok) and fail is None: fail=p
            if succ and fail: break
        if succ and fail: pairs.append((it,aid,succ,fail))
        if len(pairs)>=max_facts: break
    all_rows=[]; fc_rows=[]; fc_recov=0; fc_n=0
    for it,aid,succ,fail in pairs:
        S=_run_prompt(ctx,succ); F=_run_prompt(ctx,fail)
        cwF=rw.content_winner(F["z_final"],ctx["content_mask"])["content_winner"]
        dSa=mc.logit_decomp(S["h_final"],ctx["Wu_np"],aid)["projection"]
        dFa=mc.logit_decomp(F["h_final"],ctx["Wu_np"],aid)["projection"]
        dFc=mc.logit_decomp(F["h_final"],ctx["Wu_np"],cwF)["projection"]
        row=dict(projA_fail=dFa,projComp_fail=dFc,
                 margin_succ=mc.final_margin(S["z_final"],aid),margin_fail=mc.final_margin(F["z_final"],aid),
                 comp_more_frequent=bool(freq[cwF]>freq[aid]))
        all_rows.append(row)
        if not row["comp_more_frequent"]:                       # competitor NOT more frequent
            fc_rows.append(row); fc_n+=1
            hp=mc.patch_set_projection(F["h_final"],mc.unit(ctx["Wu_np"][aid]),dSa)
            fc_recov+=int(mc.redecode_argmax(hp,ctx["Wu_np"],ctx["content_mask"])==aid)
    def agg(rows):
        if not rows: return {}
        return {k:float(np.mean([r[k] for r in rows])) for k in ["projA_fail","projComp_fail","margin_succ","margin_fail"]}
    out=dict(n_pairs=len(pairs),frac_comp_more_frequent=float(np.mean([r["comp_more_frequent"] for r in all_rows])) if all_rows else float("nan"),
             n_freq_controlled=fc_n)
    out.update({f"all_{k}":v for k,v in agg(all_rows).items()})
    out.update({f"fc_{k}":v for k,v in agg(fc_rows).items()})
    out["fc_crosspatch_recovery"]=fc_recov/max(fc_n,1)
    return out

# ---------------- exp 4c: head-level ablation in the late band (batched) ----------------
def _batched_clean_capture_oproj(ctx, prompts, layers):
    """One batched clean pass; capture o_proj input (concatenated head outputs) at the last position
    for the given layers, plus final logits. Left-padding makes [:, -1, :] the true last token."""
    tk=ctx["tk"]; m=ctx["m"]; store={}; handles=[]
    def mk(li):
        def pre(mod,inp):
            store[li]=inp[0][:, -1, :].detach().float().detach().cpu().numpy()
        return pre
    for li in layers:
        handles.append(m.model.layers[li].self_attn.o_proj.register_forward_pre_hook(mk(li)))
    enc=tk(prompts,return_tensors="pt",padding=True,truncation=True,max_length=256).to(ctx["device"])
    with torch.no_grad(): out=m(**enc)
    for h in handles: h.remove()
    return out.logits[:, -1, :].float().detach().cpu().numpy(), store

def _batched_head_ablate(ctx, prompts, layer, head, mean_slice):
    """Batched pass with one head's slice of the o_proj input replaced by its mean at the last pos."""
    tk=ctx["tk"]; m=ctx["m"]; hd=ctx["head_dim"]; sl=slice(head*hd,(head+1)*hd)
    def pre(mod,inp):
        x=inp[0].clone(); x[:, -1, sl]=torch.tensor(mean_slice,dtype=x.dtype,device=x.device)
        return (x,)+tuple(inp[1:])
    h=m.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(pre)
    enc=tk(prompts,return_tensors="pt",padding=True,truncation=True,max_length=256).to(ctx["device"])
    with torch.no_grad(): out=m(**enc)
    h.remove()
    return out.logits[:, -1, :].float().detach().cpu().numpy()

def exp4c_head_ablation(ctx, max_items=80, batch=16, top_layers=3):
    """Localize the competitor-writing to specific heads. Restrict to the most implicated late layers
    (by aggregate competitor-lead effect), then ablate each head and rank by how much it shrinks the
    competitor's lead over the answer. Batched over failures for tractability."""
    if not ctx.get("n_heads"): return dict(status="no_head_info")
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; late=ctx["late_band"]
    # Phase A: collect losing items (answer not the content winner) + mean o_proj input per late layer
    prompts=[]; aids=[]; cws=[]; za=[]; zc=[]
    mean_acc={li:None for li in late}; cnt=0
    pool_items=[(it,fid(it["gold"][0])) for it in items]; pool_items=[(it,a) for it,a in pool_items if a is not None]
    for s in range(0,len(pool_items),batch):
        chunk=pool_items[s:s+batch]
        z,store=_batched_clean_capture_oproj(ctx,[QA.format(q=it["q"]) for it,_ in chunk],late)
        for b,(it,aid) in enumerate(chunk):
            cw=rw.content_winner(z[b],ctx["content_mask"])["content_winner"]
            if cw==aid: continue
            prompts.append(QA.format(q=it["q"])); aids.append(aid); cws.append(cw)
            za.append(float(z[b][aid])); zc.append(float(z[b][cw]))
            for li in late:
                mean_acc[li]=store[li][b].copy() if mean_acc[li] is None else mean_acc[li]+store[li][b]
            cnt+=1
        if cnt>=max_items: break
    if cnt==0: return dict(n=0)
    means={li:mean_acc[li]/cnt for li in late}
    clean_lead=np.array(zc)-np.array(aids and za)  # zc - za
    # Phase B-coarse: rank late layers by aggregate competitor-lead reduction (ablate whole o_proj input)
    layer_eff={}
    for li in late:
        dl=[]
        for s in range(0,len(prompts),batch):
            zp=_batched_head_ablate_alllayer(ctx,prompts[s:s+batch],li,means[li])
            for b in range(len(prompts[s:s+batch])):
                idx=s+b; dl.append((float(zp[b][cws[idx]])-float(zp[b][aids[idx]]))-(zc[idx]-za[idx]))
        layer_eff[li]=float(np.mean(dl))
    sel=[li for li,_ in sorted(layer_eff.items(),key=lambda kv:kv[1])[:top_layers]]   # most negative = strongest
    # Phase B-fine: per head in the selected layers
    head_rows=[]
    for li in sel:
        for hh in range(ctx["n_heads"]):
            hd=ctx["head_dim"]; mslice=means[li][hh*hd:(hh+1)*hd]
            dl=[]
            for s in range(0,len(prompts),batch):
                zp=_batched_head_ablate(ctx,prompts[s:s+batch],li,hh,mslice)
                for b in range(len(prompts[s:s+batch])):
                    idx=s+b; dl.append((float(zp[b][cws[idx]])-float(zp[b][aids[idx]]))-(zc[idx]-za[idx]))
            head_rows.append(dict(layer=li,head=hh,d_competitor_lead=float(np.mean(dl))))
    head_rows.sort(key=lambda r:r["d_competitor_lead"])   # most negative first
    return dict(n=cnt,selected_layers=sel,layer_effect={int(k):v for k,v in layer_eff.items()},
                top_heads=head_rows[:10], n_heads=ctx["n_heads"])

def _batched_head_ablate_alllayer(ctx, prompts, layer, mean_vec):
    """Mean-ablate the WHOLE o_proj input at the last position (coarse layer-level effect, batched)."""
    tk=ctx["tk"]; m=ctx["m"]
    def pre(mod,inp):
        x=inp[0].clone(); x[:, -1, :]=torch.tensor(mean_vec,dtype=x.dtype,device=x.device)
        return (x,)+tuple(inp[1:])
    h=m.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(pre)
    enc=tk(prompts,return_tensors="pt",padding=True,truncation=True,max_length=256).to(ctx["device"])
    with torch.no_grad(): out=m(**enc)
    h.remove()
    return out.logits[:, -1, :].float().detach().cpu().numpy()

# ---------------- exp 6: non-factual control task (no same-type competitor pool) ----------------
def exp_nonfactual_control(ctx, n=200, batch=16):
    """Read/write dissociation on arithmetic (no relation-specific competitor pool). If the
    dissociation is much weaker than on factual QA, the mechanism is specific to factual retrieval
    with type neighbors rather than generic to decoding."""
    tk=ctx["tk"]; m=ctx["m"]; rng=np.random.default_rng(0)
    probs=[]
    for _ in range(n):
        a=int(rng.integers(10,90)); b=int(rng.integers(10,90)); probs.append((a,b,str(a+b)))
    QAarith="Answer with a number.\nQuestion: What is {a} + {b}?\nAnswer:"
    def fid(w):
        t=tk(" "+w,add_special_tokens=False)["input_ids"]; return t[0] if t else None
    read=[]; finalread=[]; negwrite=[]; notwin=[]; ranks=[]
    items=[(a,b,ans,fid(ans)) for (a,b,ans) in probs]; items=[x for x in items if x[3] is not None]
    for s in range(0,len(items),batch):
        chunk=items[s:s+batch]
        enc=tk([QAarith.format(a=a,b=b) for a,b,_,_ in chunk],return_tensors="pt",padding=True,truncation=True,max_length=64).to(ctx["device"])
        with torch.no_grad(): out=m(**enc)
        zL=out.logits[:, -1, :].float()
        flp=torch.log_softmax(zL,-1)
        r_int=torch.full_like(flp,-1e9)
        for l in ctx["read_band"]:
            h=ctx["fn"](out.hidden_states[l][:, -1, :]).float(); r_int=torch.maximum(r_int, torch.log_softmax(h@ctx["Wu_t"].float().T,-1))
        flp=flp.detach().cpu().numpy(); rint=r_int.detach().cpu().numpy()
        for b_,(a,bb,ans,aid) in enumerate(chunk):
            read.append(float(rint[b_][aid])); finalread.append(float(flp[b_][aid]))
            rk=int((flp[b_]>flp[b_][aid]).sum()); ranks.append(rk); negwrite.append(-rk)
            notwin.append(int(rk!=0))
    return dict(n=len(items),
                rho_int=mc.spearman(read,negwrite), rho_fin=mc.spearman(finalread,negwrite),
                notwin_rate=float(np.mean(notwin)), median_rank=float(np.median(ranks)),
                mean_read=float(np.mean(read)))

# ---------------- exp 4d: attention-pattern trace of late heads (descriptive) ----------------
def exp4d_attention_pattern(ctx, max_items=120):
    """For readable failures, summarize where late-layer attention mass goes at the answer position:
    onto position 0 (BOS / sink) vs the final token vs the middle (question body), and whether
    sink-attention correlates with the competitor's lead. Descriptive."""
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; late=ctx["late_band"]
    tk=ctx["tk"]; m=ctx["m"]
    sink=[]; lastf=[]; midf=[]; lead=[]; n=0
    for it in items:
        aid=fid(it["gold"][0])
        if aid is None: continue
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]): continue
        enc=tk(p,return_tensors="pt",truncation=True,max_length=256).to(ctx["device"])
        with torch.no_grad(): out=m(**enc,output_attentions=True)
        z=out.logits[0,-1,:].float().detach().cpu().numpy()
        cw=rw.content_winner(z,ctx["content_mask"])["content_winner"]
        if cw==aid: continue
        T=enc["input_ids"].shape[1]
        # average attention (over heads) at the last query position, for late layers
        amass=np.zeros(T)
        for li in late:
            A=out.attentions[li][0,:,-1,:].float().detach().cpu().numpy()   # (heads, T)
            amass+=A.mean(0)
        amass/=max(len(late),1)
        sink.append(float(amass[0])); lastf.append(float(amass[-1])); midf.append(float(amass[1:-1].sum()))
        lead.append(float(z[cw]-z[aid])); n+=1
        if n>=max_items: break
    if n==0: return dict(n=0)
    return dict(n=n, mean_sink_attn=float(np.mean(sink)), mean_last_attn=float(np.mean(lastf)),
                mean_middle_attn=float(np.mean(midf)),
                corr_sink_vs_competitor_lead=mc.spearman(sink,lead))


# ================= HARDENING EXPERIMENTS =================

def _freq_direction(Wu_np, freq):
    """r = unit( sum_w (f_w - fbar) W_U[w] ) — the directional frequency axis of the unembedding."""
    f=np.asarray(freq,np.float64); fbar=f.mean()
    r=((f-fbar)[:,None]*Wu_np).sum(0)
    n=np.linalg.norm(r); return r/n if n>0 else r

# ---------------- exp S1: non-circular static-feature selection regression ----------------
def exp_selection_static(ctx, freq, sem_sim, label_text, max_items=400, cand_k=12, freqmatch_tol=1.0):
    """Among a candidate set {gold, selected, same-relation decoys, frequency-matched decoys}, predict
    which candidate gets the top final logit using STATIC features only: d_logfreq, same_type,
    sem_sim(gold,cand), unembed_cos(gold,cand). No final_logit, no projection (both circular). The
    clean result: frequency/geometry predict selection; external semantic similarity adds little."""
    import semreg_core as sr
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; Wu=ctx["Wu_np"]; mask=ctx["content_mask"]
    content_ids=np.where(mask)[0]
    rng=np.random.default_rng(0); sel_items=[]
    def cos_u(a,b):
        va=Wu[a]; vb=Wu[b]; na_=np.linalg.norm(va); nb=np.linalg.norm(vb)
        return float(np.dot(va,vb)/(na_*nb)) if na_>0 and nb>0 else 0.0
    pool_items=[(it,fid(it["gold"][0])) for it in items]; pool_items=[(it,a) for it,a in pool_items if a is not None]
    for it,aid in pool_items:
        R=_run_prompt(ctx,QA.format(q=it["q"]))
        cw=rw.content_winner(R["z_final"],mask)["content_winner"]
        if cw==aid: continue
        same=[i for i in ctx["relpool"][it["rel"]] if i!=aid]
        # frequency-matched (possibly cross-relation) decoys near the gold frequency -> makes same_type vary
        fcand=content_ids[np.abs(freq[content_ids]-freq[aid])<=freqmatch_tol]
        fmatch=list(rng.choice(fcand,min(4,len(fcand)),replace=False)) if len(fcand) else []
        cands=list(dict.fromkeys([aid,cw]+same[:6]+list(fmatch)))
        z=R["z_final"]; flp=z-(z.max()+np.log(np.exp(z-z.max()).sum()))
        cands=sorted(cands,key=lambda c:-flp[c])[:cand_k]
        if len(cands)<4: continue
        gtxt=label_text(aid); winner=max(cands,key=lambda c:flp[c])
        feats=[]; widx=None
        for ci,c in enumerate(cands):
            ss=sem_sim(gtxt,label_text(c))
            feats.append(dict(d_logfreq=float(freq[c]-freq[aid]),
                              same_type=float(c in set(ctx["relpool"][it["rel"]])),
                              sem_sim=float(ss) if np.isfinite(ss) else 0.0,
                              unembed_cos=cos_u(aid,c)))
            if c==winner: widx=ci
        if widx is None: continue
        sel_items.append(dict(cands=feats,winner_idx=widx))
        if len(sel_items)>=max_items: break
    feats=["d_logfreq","same_type","sem_sim","unembed_cos"]
    fit=sr.fit_selection(sel_items,l2=1.0,features=feats)
    return dict(n_items=len(sel_items),**{f:fit.get(f) for f in feats}) if "d_logfreq" in fit else dict(n_items=len(sel_items),status=fit.get("status"))

# ---------------- exp S2: causal frequency-direction intervention ----------------
def exp_freq_direction_ablation(ctx, freq, max_items=200):
    """Ablate the residual's projection onto the frequency direction r (and a random-direction
    control), and measure: does the selected token change, does the gold recover, does the winner's
    frequency drop (high-freq competitors demoted)?"""
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; Wu=ctx["Wu_np"]; mask=ctx["content_mask"]
    r=_freq_direction(Wu,freq); rng=np.random.default_rng(0)
    def content_argmax(z): 
        zz=np.where(mask,z,-np.inf); return int(np.argmax(zz))
    res={"freq":dict(changed=0,recovered=0,dfreq=[]), "random":dict(changed=0,recovered=0,dfreq=[])}; n=0
    pool_items=[(it,fid(it["gold"][0])) for it in items]; pool_items=[(it,a) for it,a in pool_items if a is not None]
    for it,aid in pool_items:
        R=_run_prompt(ctx,QA.format(q=it["q"])); h=R["h_final"]
        old=rw.content_winner(R["z_final"],mask)["content_winner"]
        if old==aid: continue
        # frequency-direction ablation: remove the residual's projection onto r
        h_f=h-np.dot(h,r)*r
        z_f=Wu@h_f; new_f=content_argmax(z_f)
        res["freq"]["changed"]+=int(new_f!=old); res["freq"]["recovered"]+=int(new_f==aid)
        res["freq"]["dfreq"].append(float(freq[new_f]-freq[old]))
        # random-direction control: matched projection magnitude removed along a random unit vector
        ru=rng.standard_normal(h.shape[0]); ru/=np.linalg.norm(ru)
        h_r=h-np.dot(h,ru)*ru
        z_r=Wu@h_r; new_r=content_argmax(z_r)
        res["random"]["changed"]+=int(new_r!=old); res["random"]["recovered"]+=int(new_r==aid)
        res["random"]["dfreq"].append(float(freq[new_r]-freq[old]))
        n+=1
        if n>=max_items: break
    out=dict(n=n)
    for k in ["freq","random"]:
        out[f"{k}_changed_rate"]=res[k]["changed"]/max(n,1)
        out[f"{k}_recovered_rate"]=res[k]["recovered"]/max(n,1)
        out[f"{k}_mean_winner_dfreq"]=float(np.mean(res[k]["dfreq"])) if res[k]["dfreq"] else float("nan")
    return out

# ---------------- exp S3: read-measure robustness sweep ----------------
def exp_read_robustness(ctx, freq, max_items=200, n_decoys=12):
    """Recompute the dissociation signal under variants of the read definition: layer band,
    max-vs-mean aggregation, mean-vs-hard decoy criterion, and frequency-matched vs random decoys.
    Reports rho_int, rho_fin, the gap, and the readable fraction per variant; stability across
    variants is the robustness claim."""
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; Wu_t=ctx["Wu_t"]; fn=ctx["fn"]; L=ctx["L"]; mask=ctx["content_mask"]
    rng=np.random.default_rng(0)
    recs=[]
    pool_items=[(it,fid(it["gold"][0])) for it in items]; pool_items=[(it,a) for it,a in pool_items if a is not None]
    for it,aid in pool_items:
        tk=ctx["tk"]; enc=tk(QA.format(q=it["q"]),return_tensors="pt",truncation=True,max_length=256).to(ctx["device"])
        with torch.no_grad(): out=ctx["m"](**enc)
        hs=out.hidden_states
        # per-layer logprob of the answer (all layers)
        alp=[]
        for l in range(L+1):
            h=fn(hs[l][0,-1,:]).float(); z=h@Wu_t.float().T; alp.append(float(torch.log_softmax(z,-1)[aid].detach().cpu()))
        # decoys (same-relation), with per-layer logprob + frequency
        pool=[i for i in ctx["relpool"][it["rel"]] if i!=aid]
        dk=list(rng.choice(pool,min(n_decoys,len(pool)),replace=False)) if pool else []
        dlp={d:[] for d in dk}
        if dk:
            for l in range(L+1):
                h=fn(hs[l][0,-1,:]).float(); lp=torch.log_softmax(h@Wu_t.float().T,-1).detach().cpu().numpy()
                for d in dk: dlp[d].append(float(lp[d]))
        zf=out.logits[0,-1,:].float().detach().cpu().numpy()
        write_rank=int((zf>zf[aid]).sum())
        recs.append(dict(alp=np.array(alp), dlp={d:np.array(v) for d,v in dlp.items()},
                         dfreq={d:float(freq[d]) for d in dk}, afreq=float(freq[aid]),
                         write_rank=write_rank, fin_lp=float(alp[L])))
        if len(recs)>=max_items: break
    # variants
    bands={"0.3-0.7":(0.3,0.7),"0.4-0.9":(0.4,0.9),"0.5-0.95":(0.5,0.95)}
    rows=[]
    for bname,(lo,hi) in bands.items():
        bl=[l for l in range(L+1) if lo<=l/L<=hi]
        for agg in ["max","mean"]:
            for crit in ["mean","hard"]:
                for dec in ["all","freqmatch"]:
                    read=[]; negwrite=[]; finread=[]; readable_flags=[]; losing=[]
                    for rr in recs:
                        a_read=(rr["alp"][bl].max() if agg=="max" else rr["alp"][bl].mean())
                        read.append(a_read); finread.append(rr["fin_lp"]); negwrite.append(-rr["write_rank"])
                        # decoy subset
                        ds=list(rr["dlp"].keys())
                        if dec=="freqmatch": ds=[d for d in ds if abs(rr["dfreq"][d]-rr["afreq"])<=1.0]
                        if ds:
                            dvals=[(rr["dlp"][d][bl].max() if agg=="max" else rr["dlp"][d][bl].mean()) for d in ds]
                            margin=(a_read-np.mean(dvals)) if crit=="mean" else (a_read-np.max(dvals))
                        else: margin=np.nan
                        readable_flags.append(bool(np.isfinite(margin) and margin>0))
                        losing.append(rr["write_rank"]!=0)
                    read=np.array(read); negwrite=np.array(negwrite); finread=np.array(finread)
                    rho_int=mc.spearman(read,negwrite); rho_fin=mc.spearman(finread,negwrite)
                    los=np.array(losing); rdf=np.array(readable_flags)
                    readable_frac=float(rdf[los].mean()) if los.sum() else float("nan")
                    rows.append(dict(band=bname,agg=agg,decoy_crit=crit,decoys=dec,
                                     rho_int=float(rho_int),rho_fin=float(rho_fin),
                                     gap=float(rho_fin-rho_int),readable_frac_of_losing=readable_frac))
    return dict(n=len(recs),variants=rows)


# ================= PAPER-COMPLETION EXPERIMENTS =================

# ---------------- exp P1: single-token / prefix robustness ----------------
def exp_single_token_robustness(ctx, max_items=400):
    """Per tokenizer: fraction of gold answers that are single-token; fraction whose first content
    token is NOT shared by a same-relation decoy (prefix-unique); and the read/write dissociation
    restricted to (a) single-token answers and (b) prefix-unique answers. Protects 'fact-specific'."""
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; tk=ctx["tk"]; mask=ctx["content_mask"]
    enc1=lambda s: tk(s,add_special_tokens=False)["input_ids"]
    n=0; n_single=0; n_unique=0
    read=[]; negwrite=[]; single_flag=[]; unique_flag=[]
    pool=[(it,fid(it["gold"][0])) for it in items]; pool=[(it,a) for it,a in pool if a is not None]
    for it,aid in pool:
        single=mc.is_single_token(it["gold"][0], enc1)
        alias_ids=[fid(a) for a in it["gold"]]
        decoy_ids=[fid(a) for a in set(REL_lookup(ctx,it["rel"])) if fid(a)!=aid] if False else \
                  [d for d in ctx["relpool"][it["rel"]] if d!=aid]
        unique=mc.first_token_unique(aid, alias_ids, decoy_ids)
        R=_run_prompt(ctx,QA.format(q=it["q"]))
        rk=int((R["z_final"]>R["z_final"][aid]).sum())
        read.append(float(R["r_int"][aid])); negwrite.append(-rk)
        single_flag.append(single); unique_flag.append(unique)
        n+=1; n_single+=int(single); n_unique+=int(unique)
        if n>=max_items: break
    read=np.array(read); negwrite=np.array(negwrite); sf=np.array(single_flag); uf=np.array(unique_flag)
    def rho(mask_):
        if mask_.sum()<10: return float("nan")
        return mc.spearman(read[mask_],negwrite[mask_])
    return dict(n=n, frac_single_token=n_single/max(n,1), frac_prefix_unique=n_unique/max(n,1),
                rho_int_all=mc.spearman(read,negwrite),
                rho_int_single=rho(sf), rho_int_unique=rho(uf),
                rho_int_single_and_unique=rho(sf&uf))

def REL_lookup(ctx, rel):  # small shim so the function above is self-contained if REL not global
    return []

# ---------------- exp P3: calibrated answer-up / alternative-down dose-response ----------------
def exp_calibrated_doseresponse(ctx, max_items=200):
    """Explicit targets from the successful-item answer-projection distribution (25/50/75th pct +
    mean), and dose-response recovery curves for answer-up, alternative-down, both, and random
    controls. Turns 'we pushed the answer' into a calibrated counterfactual."""
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; Wu=ctx["Wu_np"]; mask=ctx["content_mask"]
    # successful-item answer projection distribution
    succ=[]
    for it in items[:200]:
        aid=fid(it["gold"][0])
        if aid is None: continue
        p=QA.format(q=it["q"])
        if not _match(_greedy(ctx,p),it["gold"]): continue
        R=_run_prompt(ctx,p); succ.append(mc.logit_decomp(R["h_final"],Wu,aid)["projection"])
    if not succ: return dict(n=0, status="no_successes")
    succ=np.array(succ)
    targets={"p25":float(np.percentile(succ,25)),"p50":float(np.percentile(succ,50)),
             "p75":float(np.percentile(succ,75)),"mean":float(succ.mean())}
    # collect readable failures
    fails=[]
    for it in items:
        aid=fid(it["gold"][0])
        if aid is None: continue
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]): continue
        R=_run_prompt(ctx,p); cw=rw.content_winner(R["z_final"],mask)["content_winner"]
        if cw==aid: continue
        fails.append((R["h_final"],aid,cw))
        if len(fails)>=max_items: break
    def recov(hp_fn):
        if not fails: return float("nan")
        c=0
        for h,aid,cw in fails:
            hp=hp_fn(h,aid,cw); 
            z=np.where(mask,Wu@hp,-np.inf); c+=int(int(np.argmax(z))==aid)
        return c/len(fails)
    curves={"answer_up":{},"alternative_down":{},"both":{},"random":{}}
    rng=np.random.default_rng(0)
    for tname,T in targets.items():
        curves["answer_up"][tname]=recov(lambda h,aid,cw,T=T: mc.patch_set_projection(h,mc.unit(Wu[aid]),T))
        curves["alternative_down"][tname]=recov(lambda h,aid,cw,T=T: mc.patch_subtract(h,mc.unit(Wu[cw]),abs(T)))
        curves["both"][tname]=recov(lambda h,aid,cw,T=T: mc.patch_subtract(mc.patch_set_projection(h,mc.unit(Wu[aid]),T),mc.unit(Wu[cw]),abs(T)))
        curves["random"][tname]=recov(lambda h,aid,cw,T=T: mc.patch_subtract(h,mc.unit(rng.standard_normal(h.shape[0])),abs(T)))
    return dict(n=len(fails), n_success=len(succ), targets=targets, curves=curves)

# ---------------- exp P8: sequence-level sanity (first-token recovery -> correct continuation) ----------------
def exp_sequence_sanity(ctx, max_items=120):
    """For readable failures, patch the answer projection to the median successful target and then
    GENERATE; check whether first-token recovery actually yields the correct full answer under greedy
    decoding. Tests that the first-token analysis is not merely token-local."""
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; Wu=ctx["Wu_np"]; mask=ctx["content_mask"]; tk=ctx["tk"]; m=ctx["m"]; L=ctx["L"]
    succ=[]
    for it in items[:150]:
        aid=fid(it["gold"][0])
        if aid is None: continue
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]): continue
    # target from successes
    for it in items[:150]:
        aid=fid(it["gold"][0])
        if aid is None: continue
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]):
            R=_run_prompt(ctx,p); succ.append(mc.logit_decomp(R["h_final"],Wu,aid)["projection"])
    T=float(np.median(succ)) if succ else 10.0
    first_recovered=0; full_correct=0; n=0
    ua_cache={}
    for it in items:
        aid=fid(it["gold"][0])
        if aid is None: continue
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]): continue
        # patch the LAST-layer residual at the final position during a short generation via a hook
        ua=mc.unit(Wu[aid]); ua_t=None
        enc=tk(p,return_tensors="pt",truncation=True,max_length=256).to(ctx["device"])
        ua_t=torch.tensor(ua,dtype=torch.float32,device=ctx["device"])
        def hook(mod,inp,out):
            o=out[0] if isinstance(out,tuple) else out
            h=ctx["fn"](o[0,-1,:]).float()
            proj=torch.dot(h,ua_t); h2=h-proj*ua_t+T*ua_t
            # write back is approximate (we cannot invert final-norm exactly); apply on the pre-norm
            o[0,-1,:]=o[0,-1,:]+ (T-float(proj))*torch.tensor(ua,dtype=o.dtype,device=o.device)
            return (o,)+tuple(out[1:]) if isinstance(out,tuple) else o
        hd=m.model.layers[L-1].register_forward_hook(hook)
        with torch.no_grad():
            g=m.generate(**enc,max_new_tokens=12,do_sample=False,num_beams=1,pad_token_id=tk.pad_token_id)
        hd.remove()
        gen=tk.decode(g[0,enc["input_ids"].shape[1]:],skip_special_tokens=True)
        first_tok=g[0,enc["input_ids"].shape[1]].item() if g.shape[1]>enc["input_ids"].shape[1] else -1
        first_recovered+=int(first_tok==aid); full_correct+=int(_match(gen,it["gold"])); n+=1
        if n>=max_items: break
    return dict(n=n, target=T, first_token_recovered_rate=first_recovered/max(n,1),
                full_answer_correct_rate=full_correct/max(n,1),
                conditional_full_given_first=(full_correct/max(first_recovered,1)) if first_recovered else float("nan"))

# ---------------- exp P6: hard-decoy headline (mean vs max decoy readability + not-top rate) ----------------
def exp_harddecoy_headline(ctx, max_items=400):
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; mask=ctx["content_mask"]
    mean_read=0; hard_read=0; nfail=0; nottop_mean=0; nottop_hard=0
    rng=np.random.default_rng(7)
    for it in items:
        aid=fid(it["gold"][0])
        if aid is None: continue
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]): continue
        R=_run_prompt(ctx,p); ra=float(R["r_int"][aid])
        pool=[i for i in ctx["relpool"][it["rel"]] if i!=aid]
        dk=list(rng.choice(pool,min(8,len(pool)),replace=False)) if pool else []
        if not dk: continue
        rd=[float(R["r_int"][d]) for d in dk]; nfail+=1
        rk=int((R["z_final"]>R["z_final"][aid]).sum()); nottop=int(rk!=0)
        if mc.priming_margin_mean(ra,rd)>0: mean_read+=1; nottop_mean+=nottop
        if mc.priming_margin_hard(ra,rd)>0: hard_read+=1; nottop_hard+=nottop
        if nfail>=max_items: break
    return dict(n=nfail,
                readable_mean=mean_read/max(nfail,1), readable_hard=hard_read/max(nfail,1),
                nottop_rate_mean_readable=(nottop_mean/max(mean_read,1)) if mean_read else float("nan"),
                nottop_rate_hard_readable=(nottop_hard/max(hard_read,1)) if hard_read else float("nan"))


# ---------------- exp P4: per-model intervention heterogeneity vs trajectory category ----------------
def exp_intervention_heterogeneity(ctx, max_items=200, target="p50"):
    """Tie recovery to trajectory structure: per model, the alternative-dominant rate (from the
    layerwise trajectory classifier) alongside answer-up / alternative-down / both recovery at a fixed
    calibrated target. Explains models like Qwen2.5-3B where answer-up alone fails, both works, and the
    model is strongly alternative-dominant."""
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; Wu=ctx["Wu_np"]; mask=ctx["content_mask"]
    # calibrated target from successful answer projections
    succ=[]
    for it in items[:200]:
        aid=fid(it["gold"][0])
        if aid is None: continue
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]):
            R=_run_prompt(ctx,p); succ.append(mc.logit_decomp(R["h_final"],Wu,aid)["projection"])
    if not succ: return dict(n=0, status="no_successes")
    Tmap={"p25":np.percentile(succ,25),"p50":np.percentile(succ,50),"p75":np.percentile(succ,75),"mean":np.mean(succ)}
    T=float(Tmap[target])
    fails=[]; traj=[]
    for it in items:
        aid=fid(it["gold"][0])
        if aid is None: continue
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]): continue
        R=_run_prompt(ctx,p,want_alllayers=True); cw=rw.content_winner(R["z_final"],mask)["content_winner"]
        if cw==aid: continue
        fails.append((R["h_final"],aid,cw))
        # trajectory class from the layerwise answer-vs-competitor rank traces
        tr=mc.layerwise_trace(R["z_layers"],[aid,cw])
        cls=mc.classify_trajectory(tr[int(aid)]["rank"], tr[int(cw)]["rank"])
        traj.append(cls)
        if len(fails)>=max_items: break
    def recov(fn):
        if not fails: return float("nan")
        c=0
        for h,aid,cw in fails:
            z=np.where(mask,Wu@fn(h,aid,cw),-np.inf); c+=int(int(np.argmax(z))==aid)
        return c/len(fails)
    rng=np.random.default_rng(0)
    au=recov(lambda h,aid,cw: mc.patch_set_projection(h,mc.unit(Wu[aid]),T))
    ad=recov(lambda h,aid,cw: mc.patch_subtract(h,mc.unit(Wu[cw]),abs(T)))
    bo=recov(lambda h,aid,cw: mc.patch_subtract(mc.patch_set_projection(h,mc.unit(Wu[aid]),T),mc.unit(Wu[cw]),abs(T)))
    rnd=recov(lambda h,aid,cw: mc.patch_subtract(h,mc.unit(rng.standard_normal(h.shape[0])),abs(T)))
    cls=[t for t in traj if t]; 
    alt_dom=float(np.mean([t=="competitor_dominant" for t in cls])) if cls else float("nan")
    return dict(n=len(fails), target=target, target_value=T,
                alternative_dominant_rate=alt_dom,
                answer_up_recovery=au, alternative_down_recovery=ad,
                both_recovery=bo, random_recovery=rnd,
                synergy=float(bo-max(au,ad)))   # positive synergy = both exceeds either alone


# ================= VALIDITY / SCOPE EXPERIMENTS =================

# ---------------- exp V1: full-answer continuation after first-token patch (A/B/C/D decomposition) ----------------
def exp_continuation_decomposition(ctx, max_items=120, target_pct=50):
    """After patching the answer direction at the FIRST generated step only, greedy-continue normally
    and classify each readable failure:
      A first-token recovered AND continuation completes the gold alias
      B first-token recovered but continuation drifts (wrong rest)
      C first-token recovered but alias matching fails (right tokens, normalization mismatch)
      D first token was a weak prefix (gold not single-token; first token shared with a non-gold)
    Also reports teacher-forced logprob of the remaining gold tokens after the patched first token.
    If B dominates, the paper's scope should stay first-token selection."""
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; Wu=ctx["Wu_np"]; mask=ctx["content_mask"]
    tk=ctx["tk"]; m=ctx["m"]; L=ctx["L"]
    enc1=lambda s: tk(s,add_special_tokens=False)["input_ids"]
    # calibrated target from successes
    succ=[]
    for it in items[:200]:
        aid=fid(it["gold"][0])
        if aid is None: continue
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]):
            R=_run_prompt(ctx,p); succ.append(mc.logit_decomp(R["h_final"],Wu,aid)["projection"])
    T=float(np.percentile(succ,target_pct)) if succ else 12.0
    cats={"A":0,"B":0,"C":0,"D":0,"first_not_recovered":0}; n=0; tf_rest=[]
    for it in items:
        aid=fid(it["gold"][0])
        if aid is None: continue
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]): continue
        gold0=it["gold"][0]; gold_ids=enc1(" "+gold0)
        single=len(gold_ids)==1
        ua=mc.unit(Wu[aid]); ua_o=torch.tensor(ua,dtype=torch.float32,device=ctx["device"])
        enc=tk(p,return_tensors="pt",truncation=True,max_length=256).to(ctx["device"])
        # patch ONLY the first generated step: additive nudge along answer dir at last layer, first step
        state={"step":0}
        def hook(mod,inp,out):
            o=out[0] if isinstance(out,tuple) else out
            if state["step"]==0:
                h=ctx["fn"](o[0,-1,:]).float(); proj=float(torch.dot(h,ua_o))
                o[0,-1,:]=o[0,-1,:]+(T-proj)*torch.tensor(ua,dtype=o.dtype,device=o.device)
            state["step"]+=1
            return (o,)+tuple(out[1:]) if isinstance(out,tuple) else o
        hd=m.model.layers[L-1].register_forward_hook(hook)
        with torch.no_grad():
            g=m.generate(**enc,max_new_tokens=12,do_sample=False,num_beams=1,pad_token_id=tk.pad_token_id)
        hd.remove()
        new_ids=g[0,enc["input_ids"].shape[1]:].tolist()
        first_tok=new_ids[0] if new_ids else -1
        gen=tk.decode(new_ids,skip_special_tokens=True)
        # teacher-forced logprob of remaining gold tokens given the patched first token
        if len(gold_ids)>1:
            enc2=tk(p+ " "+gold0,return_tensors="pt",truncation=True,max_length=300).to(ctx["device"])
            with torch.no_grad(): o2=m(**enc2)
            lp=torch.log_softmax(o2.logits[0],-1)
            start=enc["input_ids"].shape[1]
            rest=[float(lp[start+i-1, gold_ids[i]]) for i in range(1,len(gold_ids)) if start+i-1 < lp.shape[0]]
            if rest: tf_rest.append(float(np.mean(rest)))
        if first_tok!=aid:
            cats["first_not_recovered"]+=1; n+=1
            if n>=max_items: break
            continue
        # first token recovered: classify A/B/C/D
        if not single and (first_tok in [d for d in ctx["relpool"][it["rel"]] if d!=aid]):
            cats["D"]+=1                                   # weak prefix shared with a decoy
        elif _match(gen,it["gold"]):
            cats["A"]+=1                                   # full alias completed
        else:
            # right first token, wrong rest: B (drift) vs C (alias normalization) -- check normalized contains
            gnorm=_na(gen); anyalias=any(_na(a) in gnorm or gnorm in _na(a) for a in it["gold"])
            cats["C" if anyalias else "B"]+=1
        n+=1
        if n>=max_items: break
    tot=max(n,1)
    return dict(n=n, target=T,
                A_first_and_full=cats["A"]/tot, B_first_then_drift=cats["B"]/tot,
                C_alias_mismatch=cats["C"]/tot, D_weak_prefix=cats["D"]/tot,
                first_not_recovered=cats["first_not_recovered"]/tot,
                tf_rest_mean_logprob=float(np.mean(tf_rest)) if tf_rest else float("nan"),
                conclusion=("scope=first-token (B dominates)" if cats["B"]>=max(cats["A"],cats["C"],cats["D"])
                            else "continuation often completes (A competitive)"))

def _na(s):
    import re as _re
    s=s.lower().strip(); s=_re.sub(r"\b(a|an|the)\b"," ",s); s=_re.sub(r"[^a-z0-9 ]"," ",s); return _re.sub(r"\s+"," ",s).strip()

# ---------------- exp V2: multi-token sequence read/write (1-3 token answers) ----------------
def exp_sequence_readwrite(ctx, max_items=200, max_ans_tokens=3):
    """Extend read/write to short answer STRINGS (1-3 tokens). sequence_read = mean logit-lens logprob
    of gold tokens over the read band (teacher-forced); sequence_write = greedy continuation matches an
    alias; margin = gold sequence logprob - selected-alternative sequence logprob. Reports the
    read/write correlation on the multi-token subset."""
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; tk=ctx["tk"]; m=ctx["m"]; L=ctx["L"]; Wu_t=ctx["Wu_t"]; fn=ctx["fn"]; mask=ctx["content_mask"]
    enc1=lambda s: tk(s,add_special_tokens=False)["input_ids"]
    seq_read=[]; negwrite=[]; gen_correct=[]; n=0
    for it in items:
        gold0=it["gold"][0]; gold_ids=enc1(" "+gold0)
        if not (1<=len(gold_ids)<=max_ans_tokens): continue
        p=QA.format(q=it["q"])
        # teacher-forced per-layer logprob of each gold token, averaged over read band then over tokens
        enc=tk(p+" "+gold0,return_tensors="pt",truncation=True,max_length=300).to(ctx["device"])
        with torch.no_grad(): out=m(**enc)
        hs=out.hidden_states; start=tk(p,return_tensors="pt",truncation=True,max_length=256)["input_ids"].shape[1]
        per_tok=[]
        for i,gt in enumerate(gold_ids):
            pos=start+i-1
            if pos<0 or pos>=hs[0].shape[1]: continue
            band_lp=[]
            for l in ctx["read_band"]:
                h=fn(hs[l][0,pos,:]).float(); band_lp.append(float(torch.log_softmax(h@Wu_t.float().T,-1)[gt].detach().cpu()))
            per_tok.append(max(band_lp) if band_lp else np.nan)
        if not per_tok: continue
        sr=float(np.nanmean(per_tok)); seq_read.append(sr)
        # write: greedy continuation correctness + rank proxy (final-token rank of gold first token)
        gen=_greedy(ctx,p); ok=_match(gen,it["gold"]); gen_correct.append(int(ok))
        Rf=_run_prompt(ctx,p); rk=int((Rf["z_final"]>Rf["z_final"][gold_ids[0]]).sum()); negwrite.append(-rk)
        n+=1
        if n>=max_items: break
    return dict(n=n,
                rho_seqread_negwrite=mc.spearman(seq_read,negwrite),
                gen_correct_rate=float(np.mean(gen_correct)) if gen_correct else float("nan"),
                mean_seq_read=float(np.mean(seq_read)) if seq_read else float("nan"))

# ---------------- exp V3: stricter prefix-unique sweep + dissociation under each ----------------
def exp_prefix_unique_strict(ctx, max_items=400):
    """Recompute prefix-uniqueness under stricter criteria (unique among decoys, unique first-2 tokens,
    long-enough first token) and the dissociation correlation restricted to each subset. Verifies the
    frac_prefix_unique=1.0 result is not a tokenization artifact."""
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; tk=ctx["tk"]
    enc1=lambda s: tk(s,add_special_tokens=False)["input_ids"]
    read=[]; negwrite=[]; flags={"u1":[], "u2":[], "long":[], "strict":[]}
    counts={"u1":0,"u2":0,"long":0,"strict":0}; n=0
    pool=[(it,fid(it["gold"][0])) for it in items]; pool=[(it,a) for it,a in pool if a is not None]
    for it,aid in pool:
        gold_ids=enc1(" "+it["gold"][0]); g2=gold_ids[:2]
        decoys=[d for d in ctx["relpool"][it["rel"]] if d!=aid]
        # decoy first-2 token ids (approx: re-encode each alias label is costly; use single-id pairs from pool)
        dec2=[[d] for d in decoys]   # first-2 unavailable cheaply -> treat decoy as length-1 (conservative for u2)
        r=mc.stricter_prefix_unique([aid], g2, decoys, dec2, aid, min_char_len=3, decode=lambda ids: tk.decode(ids))
        R=_run_prompt(ctx,QA.format(q=it["q"]))
        rk=int((R["z_final"]>R["z_final"][aid]).sum())
        read.append(float(R["r_int"][aid])); negwrite.append(-rk)
        flags["u1"].append(r["unique1_decoy"]); flags["u2"].append(r["unique2_decoy"])
        flags["long"].append(r["long_enough"]); flags["strict"].append(r["strict"])
        for k in counts: counts[k]+=int(r[{"u1":"unique1_decoy","u2":"unique2_decoy","long":"long_enough","strict":"strict"}[k]])
        n+=1
        if n>=max_items: break
    read=np.array(read); negwrite=np.array(negwrite)
    def rho(flagname):
        f=np.array(flags[flagname])
        return mc.bootstrap_ci_spearman(read[f],negwrite[f]) if f.sum()>=10 else dict(point=float("nan"),lo=float("nan"),hi=float("nan"),n=int(f.sum()))
    return dict(n=n, frac_unique1_decoy=counts["u1"]/max(n,1), frac_unique2=counts["u2"]/max(n,1),
                frac_long_enough=counts["long"]/max(n,1), frac_strict=counts["strict"]/max(n,1),
                rho_all=mc.bootstrap_ci_spearman(read,negwrite),
                rho_unique1=rho("u1"), rho_long=rho("long"), rho_strict=rho("strict"))

# ---------------- exp V4: single-token subset metrics WITH bootstrap CIs ----------------
def exp_single_token_ci(ctx, max_items=400):
    """The single-token subset is small; attach bootstrap CIs to rho_single, hard-readable fraction,
    not-top rate, and answer-up recovery so wide uncertainty is reported honestly."""
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; Wu=ctx["Wu_np"]; mask=ctx["content_mask"]; tk=ctx["tk"]
    enc1=lambda s: tk(s,add_special_tokens=False)["input_ids"]
    read=[]; negwrite=[]; single=[]; hard_readable=[]; nottop=[]; recov=[]
    # target for recovery
    succ=[]
    for it in items[:200]:
        aid=fid(it["gold"][0])
        if aid is None: continue
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]):
            R=_run_prompt(ctx,p); succ.append(mc.logit_decomp(R["h_final"],Wu,aid)["projection"])
    T=float(np.median(succ)) if succ else 12.0
    rng=np.random.default_rng(0)
    pool=[(it,fid(it["gold"][0])) for it in items]; pool=[(it,a) for it,a in pool if a is not None]
    for it,aid in pool:
        is_single=len(enc1(" "+it["gold"][0]))==1
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]): continue
        R=_run_prompt(ctx,p); cw=rw.content_winner(R["z_final"],mask)["content_winner"]
        rk=int((R["z_final"]>R["z_final"][aid]).sum())
        read.append(float(R["r_int"][aid])); negwrite.append(-rk); single.append(is_single)
        pool_d=[i for i in ctx["relpool"][it["rel"]] if i!=aid]
        dk=list(rng.choice(pool_d,min(8,len(pool_d)),replace=False)) if pool_d else []
        if dk:
            hard_readable.append(int(mc.priming_margin_hard(float(R["r_int"][aid]),[float(R["r_int"][d]) for d in dk])>0))
        nottop.append(int(rk!=0))
        if cw!=aid:
            hp=mc.patch_set_projection(R["h_final"],mc.unit(Wu[aid]),T)
            recov.append(int(mc.redecode_argmax(hp,Wu,mask)==aid))
    read=np.array(read); negwrite=np.array(negwrite); sf=np.array(single)
    out=dict(n=len(read), n_single=int(sf.sum()))
    out["rho_all"]=mc.bootstrap_ci_spearman(read,negwrite)
    out["rho_single"]=mc.bootstrap_ci_spearman(read[sf],negwrite[sf]) if sf.sum()>=10 else dict(point=float("nan"),lo=float("nan"),hi=float("nan"),n=int(sf.sum()))
    out["hard_readable_frac"]=mc.bootstrap_ci(hard_readable)
    out["nottop_rate"]=mc.bootstrap_ci(nottop)
    out["answer_up_recovery"]=mc.bootstrap_ci(recov)
    return out

# ---------------- exp V7: symmetric alternative-down calibration ----------------
def exp_symmetric_altdown(ctx, max_items=200, target_pct=50):
    """Calibrate alternative-down symmetrically: set the selected alternative's support DOWN to the
    median support that alternatives have in SUCCESSFUL generations (where the alternative loses),
    rather than subtracting an answer-derived amount. Makes answer-up and alternative-down comparable."""
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; Wu=ctx["Wu_np"]; mask=ctx["content_mask"]
    # distribution of the (eventual) competitor's support in SUCCESSFUL items
    alt_succ=[]
    for it in items[:200]:
        aid=fid(it["gold"][0])
        if aid is None: continue
        p=QA.format(q=it["q"])
        if not _match(_greedy(ctx,p),it["gold"]): continue
        R=_run_prompt(ctx,p)
        # the top non-answer content token in a success = a "losing alternative"; record its projection
        z=R["z_final"].copy(); z[aid]=-np.inf; zc=np.where(mask,z,-np.inf); alt=int(np.argmax(zc))
        alt_succ.append(mc.logit_decomp(R["h_final"],Wu,alt)["projection"])
    T_alt=float(np.percentile(alt_succ,target_pct)) if alt_succ else 0.0
    # answer-up target (for comparability)
    succ=[]
    for it in items[:200]:
        aid=fid(it["gold"][0])
        if aid is None: continue
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]):
            R=_run_prompt(ctx,p); succ.append(mc.logit_decomp(R["h_final"],Wu,aid)["projection"])
    T_ans=float(np.percentile(succ,target_pct)) if succ else 12.0
    fails=[]
    for it in items:
        aid=fid(it["gold"][0])
        if aid is None: continue
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]): continue
        R=_run_prompt(ctx,p); cw=rw.content_winner(R["z_final"],mask)["content_winner"]
        if cw==aid: continue
        fails.append((R["h_final"],aid,cw))
        if len(fails)>=max_items: break
    def recov(fn):
        if not fails: return float("nan")
        c=0
        for h,aid,cw in fails: c+=int(mc.redecode_argmax(fn(h,aid,cw),Wu,mask)==aid)
        return c/len(fails)
    au=recov(lambda h,aid,cw: mc.patch_set_projection(h,mc.unit(Wu[aid]),T_ans))
    ad=recov(lambda h,aid,cw: mc.patch_set_projection(h,mc.unit(Wu[cw]),T_alt))   # set competitor DOWN to success-level
    bo=recov(lambda h,aid,cw: mc.patch_set_projection(mc.patch_set_projection(h,mc.unit(Wu[aid]),T_ans),mc.unit(Wu[cw]),T_alt))
    return dict(n=len(fails), target_answer=T_ans, target_alt_success_level=T_alt,
                answer_up_recovery=au, alt_down_to_success_recovery=ad, both_recovery=bo)


# ================= BASELINE/CONTEXTUAL DECOMPOSITION =================

def exp_baseline_decomposition(ctx, freq, max_items=300):
    """Collect final residuals for readable failures, decompose the answer-vs-competitor margin into a
    leave-one-out readout baseline margin + an item-specific contextual margin (EXACT), classify the
    regime (answer-support-limited vs baseline-limited), and test how the baseline becomes operational
    (mean residual vs frequency direction / BOS-sink direction; baseline vs token frequency)."""
    import baseline_core as bc
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; Wu=ctx["Wu_np"]; mask=ctx["content_mask"]
    H=[]; aids=[]; cids=[]
    bos_dir=None
    # BOS/sink direction: final residual on a minimal BOS-only / empty prompt (proxy for the sink read)
    try:
        Rb=_run_prompt(ctx, QA.format(q=""))
        bos_dir=Rb["h_final"]/ (np.linalg.norm(Rb["h_final"])+1e-9)
    except Exception:
        bos_dir=None
    pool=[(it,fid(it["gold"][0])) for it in items]; pool=[(it,a) for it,a in pool if a is not None]
    for it,aid in pool:
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]): continue
        R=_run_prompt(ctx,p); cw=rw.content_winner(R["z_final"],mask)["content_winner"]
        if cw==aid: continue
        H.append(R["h_final"]); aids.append(aid); cids.append(cw)
        if len(H)>=max_items: break
    if len(H)<10: return dict(n=len(H), status="too_few")
    H=np.array(H)
    rows=bc.decompose_margin(Wu,H,aids,cids,use_loo=True)
    agg=bc.aggregate(rows)
    hbar_global=H.mean(0)
    align=bc.baseline_alignment(Wu,hbar_global,freq,bos_dir=bos_dir)
    # answer-up / both recovery + alternative-dominant fraction (reuse calibrated target)
    succ=[]
    for it in items[:200]:
        a=fid(it["gold"][0])
        if a is None: continue
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]):
            Rs=_run_prompt(ctx,p); succ.append(mc.logit_decomp(Rs["h_final"],Wu,a)["projection"])
    T=float(np.median(succ)) if succ else 12.0
    fails=list(zip(H,aids,cids))
    def recov(fn):
        c=0
        for h,a,b in fails: c+=int(mc.redecode_argmax(fn(h,a,b),Wu,mask)==a)
        return c/len(fails)
    au=recov(lambda h,a,b: mc.patch_set_projection(h,mc.unit(Wu[a]),T))
    bo=recov(lambda h,a,b: mc.patch_subtract(mc.patch_set_projection(h,mc.unit(Wu[a]),T),mc.unit(Wu[b]),abs(T)))
    return dict(n=len(H), target=T,
                mean_baseline_margin=agg["mean_baseline_margin"],
                mean_contextual_margin=agg["mean_contextual_margin"],
                mean_final_margin=agg["mean_final_margin"],
                frac_baseline_negative=agg["frac_baseline_negative"],
                frac_baseline_limited=agg["frac_baseline_limited"],
                exactness=agg["max_abs_exactness"],
                answer_up_recovery=au, both_recovery=bo,
                **align)


# ================= COMPONENT-LEVEL ATTRIBUTION & CAUSAL TESTS =================

def _collect_components(ctx, prompt):
    """Run one prompt and capture each layer's attention-block and MLP-block output at the last
    position (the residual-stream contributions), plus the embedding contribution. Returns
    X: (K, d) component outputs (pre-final-norm), names, and the final residual h (post-norm) + logits.
    Components are projected through the final norm INSIDE the runner caller via a shared scale so the
    sum-to-residual property holds approximately (RMSNorm is applied to the summed residual)."""
    m=ctx["m"]; tk=ctx["tk"]; L=ctx["L"]
    caps={}
    handles=[]
    def mk(tag):
        def hook(mod,inp,out):
            o=out[0] if isinstance(out,tuple) else out
            caps[tag]=o[0,-1,:].detach().float().cpu().numpy()
        return hook
    for l in range(L):
        handles.append(m.model.layers[l].self_attn.register_forward_hook(mk(f"attn.{l}")))
        handles.append(m.model.layers[l].mlp.register_forward_hook(mk(f"mlp.{l}")))
    enc=tk(prompt,return_tensors="pt",truncation=True,max_length=256).to(ctx["device"])
    with torch.no_grad(): out=m(**enc)
    for h in handles: h.remove()
    hs=out.hidden_states
    emb=hs[0][0,-1,:].detach().float().cpu().numpy()         # embedding (+ pos) contribution
    names=["emb"]+[f"attn.{l}" for l in range(L)]+[f"mlp.{l}" for l in range(L)]
    X=np.stack([emb]+[caps[f"attn.{l}"] for l in range(L)]+[caps[f"mlp.{l}"] for l in range(L)])  # (K,d) pre-norm
    h_pre=hs[L][0,-1,:].detach().float().cpu().numpy()       # pre-final-norm residual = sum of components
    h_final=ctx["fn"](torch.tensor(h_pre,device=ctx["device"]).unsqueeze(0)).float().detach().cpu().numpy()[0]
    # RMSNorm is x/rms*w ; apply the SAME per-row scale to each component so sum(X_scaled)=h_final (approx,
    # ignoring the learned weight which is diagonal and applied equally) -> scale = ||h_final||-consistent.
    rms=np.sqrt((h_pre**2).mean()+1e-6)
    w=ctx["fn"].weight.detach().float().cpu().numpy() if hasattr(ctx["fn"],"weight") else np.ones_like(h_pre)
    Xn=(X/rms)*w[None,:]                                     # each component through the final norm
    z_final=out.logits[0,-1,:].float().detach().cpu().numpy()
    return dict(X=Xn, names=names, h_final=h_final, z_final=z_final, h_pre=h_pre)

# ---------------- exp C1: component-level margin attribution (+ baseline/contextual split) ----------------
def exp_component_attribution(ctx, max_items=150):
    """Decompose the answer-vs-alternative margin across components, split into baseline vs contextual.
    Shows which components write the frequency-linked baseline vs answer/alternative-specific context."""
    import component_core as cc
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; Wu=ctx["Wu_np"]; mask=ctx["content_mask"]
    Xs=[]; aids=[]; cids=[]; names=None
    pool=[(it,fid(it["gold"][0])) for it in items]; pool=[(it,a) for it,a in pool if a is not None]
    for it,aid in pool:
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]): continue
        comp=_collect_components(ctx,p)
        cw=rw.content_winner(comp["z_final"],mask)["content_winner"]
        if cw==aid: continue
        Xs.append(comp["X"]); aids.append(aid); cids.append(cw); names=comp["names"]
        if len(Xs)>=max_items: break
    if len(Xs)<10: return dict(n=len(Xs), status="too_few")
    X=np.stack(Xs)                                           # (n,K,d)
    att=cc.attribute_margins(X,Wu,aids,cids,use_loo=True)
    top_alt_total=cc.rank_components(att["mean_total"],names,k=10,most_negative=True)
    top_alt_base=cc.rank_components(att["mean_baseline"],names,k=10,most_negative=True)
    top_alt_ctx =cc.rank_components(att["mean_contextual"],names,k=10,most_negative=True)
    top_ans_ctx =cc.rank_components(att["mean_contextual"],names,k=10,most_negative=False)
    return dict(n=len(Xs), names=names,
                top_alternative_total=top_alt_total,
                top_alternative_baseline=top_alt_base,
                top_alternative_contextual=top_alt_ctx,
                top_answer_contextual=top_ans_ctx,
                sum_baseline=float(att["mean_baseline"].sum()),
                sum_contextual=float(att["mean_contextual"].sum()),
                sum_total=float(att["mean_total"].sum()))

# ---------------- exp C2: causal ablation of top margin components ----------------
def exp_component_ablation(ctx, max_items=150, top_k=5):
    """Ablate (mean-replace) the top alternative-favoring components and measure the margin shift vs
    random matched components. Mean-ablation uses the across-item component mean."""
    import component_core as cc
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; Wu=ctx["Wu_np"]; mask=ctx["content_mask"]; L=ctx["L"]
    Xs=[]; aids=[]; cids=[]; names=None
    pool=[(it,fid(it["gold"][0])) for it in items]; pool=[(it,a) for it,a in pool if a is not None]
    for it,aid in pool:
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]): continue
        comp=_collect_components(ctx,p); cw=rw.content_winner(comp["z_final"],mask)["content_winner"]
        if cw==aid: continue
        Xs.append(comp["X"]); aids.append(aid); cids.append(cw); names=comp["names"]
        if len(Xs)>=max_items: break
    if len(Xs)<10: return dict(n=len(Xs), status="too_few")
    X=np.stack(Xs); n,K,d=X.shape
    att=cc.attribute_margins(X,Wu,aids,cids,use_loo=True)
    order=np.argsort(att["mean_total"])                      # most alternative-favoring first
    top_idx=list(order[:top_k]); 
    rng=np.random.default_rng(0); rand_idx=list(rng.choice(K,top_k,replace=False))
    xbar=X.mean(0)                                           # (K,d) component means
    def margins_after(ablate_idx):
        out=[]
        for i in range(n):
            Xi=X[i].copy()
            for k in ablate_idx: Xi[k]=xbar[k]               # mean-ablate component
            h=Xi.sum(0); diff=Wu[aids[i]]-Wu[cids[i]]
            out.append(float(h@diff))
        return np.array(out)
    before=np.array([float(X[i].sum(0)@(Wu[aids[i]]-Wu[cids[i]])) for i in range(n)])
    after_top=margins_after(top_idx); after_rand=margins_after(rand_idx)
    # first-token recovery + selected-token frequency shift after top-ablation
    rec=0; dfreq=[]
    for i in range(n):
        Xi=X[i].copy()
        for k in top_idx: Xi[k]=xbar[k]
        z=np.where(mask,Wu@Xi.sum(0),-np.inf); new=int(np.argmax(z))
        rec+=int(new==aids[i]); dfreq.append(float(ctx["freq"][new]-ctx["freq"][cids[i]]))
    return dict(n=n, top_components=[names[k] for k in top_idx], random_components=[names[k] for k in rand_idx],
                shift_top=cc.margin_shift(before,after_top),
                shift_random=cc.margin_shift(before,after_rand),
                first_token_recovery=rec/n, mean_selected_freq_shift=float(np.mean(dfreq)))

# ---------------- exp C3: success->failure paired-paraphrase component patching ----------------
def exp_paired_component_patch(ctx, max_facts=80, blocks=None):
    """For facts that SUCCEED under one paraphrase and FAIL under another (same fact/answer/relation/
    frequency), patch component outputs from the success into the failure and measure margin/answer-
    support/first-token recovery. The cleanest causal test: it controls for everything but the
    components patched."""
    import component_core as cc
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; T=ctx["templates"]; Wu=ctx["Wu_np"]; mask=ctx["content_mask"]; L=ctx["L"]
    if blocks is None: blocks={"late_attn":[f"attn.{l}" for l in range(int(0.75*L),L)],
                               "late_mlp":[f"mlp.{l}" for l in range(int(0.75*L),L)],
                               "late_all":[f"attn.{l}" for l in range(int(0.75*L),L)]+[f"mlp.{l}" for l in range(int(0.75*L),L)]}
    pairs=[]
    for it in items:
        aid=fid(it["gold"][0])
        if aid is None: continue
        succ=fail=None
        for tpl in T:
            p=tpl.format(q=it["q"]); ok=_match(_greedy(ctx,p),it["gold"])
            if ok and succ is None: succ=p
            if (not ok) and fail is None: fail=p
            if succ and fail: break
        if succ and fail: pairs.append((it,aid,succ,fail))
        if len(pairs)>=max_facts: break
    if len(pairs)<5: return dict(n=len(pairs), status="too_few")
    results={bn:{"before":[],"after":[],"recovered":0} for bn in blocks}
    name_index=None
    for it,aid,succ,fail in pairs:
        Cs=_collect_components(ctx,succ); Cf=_collect_components(ctx,fail)
        if name_index is None: name_index={nm:k for k,nm in enumerate(Cf["names"])}
        cwf=rw.content_winner(Cf["z_final"],mask)["content_winner"]
        if cwf==aid: continue
        diff=Wu[aid]-Wu[cwf]
        base_margin=float(Cf["X"].sum(0)@diff)
        for bn,comp_names in blocks.items():
            Xi=Cf["X"].copy()
            for nm in comp_names:
                if nm in name_index: Xi[name_index[nm]]=Cs["X"][name_index[nm]]   # patch success->failure
            h=Xi.sum(0); m_after=float(h@diff)
            results[bn]["before"].append(base_margin); results[bn]["after"].append(m_after)
            z=np.where(mask,Wu@h,-np.inf); results[bn]["recovered"]+=int(int(np.argmax(z))==aid)
    out=dict(n_pairs=len(pairs))
    for bn,r in results.items():
        if r["before"]:
            ms=cc.margin_shift(r["before"],r["after"])
            out[bn]=dict(mean_shift=ms["mean_shift"], frac_improved=ms["frac_improved"],
                         recovery=r["recovered"]/len(r["before"]), n=len(r["before"]))
    return out

# ---------------- exp C4: source of the readout baseline ----------------
def exp_baseline_source(ctx, freq, max_items=200):
    """Decompose the mean residual hbar by component: which components contribute most to hbar, and how
    much does each contribute to corr(baseline, frequency)? Ablating a component's mean tests whether
    the frequency-graded baseline is carried by specific components (e.g. late MLP / sink-reading attn)."""
    import component_core as cc
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; Wu=ctx["Wu_np"]; mask=ctx["content_mask"]; L=ctx["L"]
    Xs=[]; names=None
    pool=[(it,fid(it["gold"][0])) for it in items]; pool=[(it,a) for it,a in pool if a is not None]
    for it,aid in pool:
        comp=_collect_components(ctx,QA.format(q=it["q"]))
        Xs.append(comp["X"]); names=comp["names"]
        if len(Xs)>=max_items: break
    if len(Xs)<10: return dict(n=len(Xs), status="too_few")
    X=np.stack(Xs); xbar=X.mean(0)                          # (K,d) component means
    hbar=xbar.sum(0)
    # contribution of each component to hbar (norm) and to corr(baseline,freq)
    f=np.asarray(freq,float)
    comp_norm=np.linalg.norm(xbar,axis=1)
    # baseline from full hbar
    b_full=Wu@hbar; corr_full=float(np.corrcoef(b_full,f)[0,1]) if b_full.std()>0 else float("nan")
    # leave-one-component-out: how much corr(base,freq) drops when component k's mean is removed
    drops=[]
    for k in range(len(names)):
        hk=hbar-xbar[k]; bk=Wu@hk
        ck=float(np.corrcoef(bk,f)[0,1]) if bk.std()>0 else float("nan")
        drops.append((names[k], corr_full-ck, float(comp_norm[k])))
    drops_by_corr=sorted(drops,key=lambda t:-abs(t[1]))[:10]
    drops_by_norm=sorted(drops,key=lambda t:-t[2])[:10]
    return dict(n=len(Xs), corr_baseline_freq_full=corr_full,
                top_components_by_freqcorr_contribution=[(nm,round(d,4)) for nm,d,_ in drops_by_corr],
                top_components_by_meannorm=[(nm,round(nr,3)) for nm,_,nr in drops_by_norm])


# ================= PAIRED-PATCH CONTROLS & STABILITY =================

def exp_paired_patch_controls(ctx, max_facts=80, late_frac=0.75):
    """Four-arm control for the success->failure paired patch. For each failed paraphrase, patch LATE
    components from:
      (a) same_fact   : the successful paraphrase of the SAME fact (the real effect)
      (b) diff_fact   : a successful paraphrase of a DIFFERENT, random fact (controls 'any success helps')
      (c) rand_late   : a RANDOM subset of late components from the same success (controls 'any late patch')
      (d) early       : EARLY components from the same success (controls 'late-ness' specifically)
    Reports margin shift, first-token recovery, and answer/alternative support change for each arm."""
    import component_core as cc
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; T=ctx["templates"]; Wu=ctx["Wu_np"]; mask=ctx["content_mask"]; L=ctx["L"]
    lo=int(late_frac*L)
    late_names=[f"attn.{l}" for l in range(lo,L)]+[f"mlp.{l}" for l in range(lo,L)]
    early_names=[f"attn.{l}" for l in range(0,L-lo)]+[f"mlp.{l}" for l in range(0,L-lo)]
    # build paired (success,fail) for facts; cache success component sets for cross-fact donors
    pairs=[]
    for it in items:
        aid=fid(it["gold"][0])
        if aid is None: continue
        succ=fail=None
        for tpl in T:
            p=tpl.format(q=it["q"]); ok=_match(_greedy(ctx,p),it["gold"])
            if ok and succ is None: succ=p
            if (not ok) and fail is None: fail=p
            if succ and fail: break
        if succ and fail: pairs.append((it,aid,succ,fail))
        if len(pairs)>=max_facts: break
    if len(pairs)<6: return dict(n=len(pairs), status="too_few")
    # precompute components
    comp_cache={}
    def comps(p):
        if p not in comp_cache: comp_cache[p]=_collect_components(ctx,p)
        return comp_cache[p]
    name_index=None; rng=np.random.default_rng(0)
    arms={a:{"before":[],"after":[],"rec":0,"dsupp_ans":[],"dsupp_alt":[]} for a in ["same_fact","diff_fact","rand_late","early"]}
    succ_prompts=[s for _,_,s,_ in pairs]
    for idx,(it,aid,succ,fail) in enumerate(pairs):
        Cf=comps(fail); Cs=comps(succ)
        if name_index is None: name_index={nm:k for k,nm in enumerate(Cf["names"])}
        cwf=rw.content_winner(Cf["z_final"],mask)["content_winner"]
        if cwf==aid: continue
        diff=Wu[aid]-Wu[cwf]; base_margin=float(Cf["X"].sum(0)@diff)
        ua=mc.unit(Wu[aid]); ub=mc.unit(Wu[cwf])
        supp_ans0=float(Cf["X"].sum(0)@ua); supp_alt0=float(Cf["X"].sum(0)@ub)
        # donor for diff_fact: a different fact's success
        j=(idx+1+rng.integers(0,max(len(pairs)-1,1)))%len(pairs)
        Cdiff=comps(pairs[j][2])
        # random late subset (same size as late set, but random component indices)
        rand_late=list(rng.choice(len(Cf["names"]), len(late_names), replace=False))
        def patch(donor, comp_names=None, comp_idx=None):
            Xi=Cf["X"].copy()
            if comp_idx is not None:
                for k in comp_idx: Xi[k]=donor["X"][k]
            else:
                for nm in comp_names:
                    if nm in name_index: Xi[name_index[nm]]=donor["X"][name_index[nm]]
            h=Xi.sum(0)
            return float(h@diff), int(np.argmax(np.where(mask,Wu@h,-np.inf))), float(h@ua), float(h@ub)
        for arm,(donor,kw) in {
            "same_fact":(Cs,{"comp_names":late_names}),
            "diff_fact":(Cdiff,{"comp_names":late_names}),
            "rand_late":(Cs,{"comp_idx":rand_late}),
            "early":(Cs,{"comp_names":early_names}),
        }.items():
            m_after,newtok,supp_ans,supp_alt=patch(donor,**kw)
            A=arms[arm]; A["before"].append(base_margin); A["after"].append(m_after); A["rec"]+=int(newtok==aid)
            A["dsupp_ans"].append(supp_ans-supp_ans0); A["dsupp_alt"].append(supp_alt-supp_alt0)
    out=dict(n_pairs=len(pairs))
    for arm,A in arms.items():
        if A["before"]:
            ms=cc.margin_shift(A["before"],A["after"])
            out[arm]=dict(mean_shift=ms["mean_shift"], frac_improved=ms["frac_improved"],
                          recovery=A["rec"]/len(A["before"]),
                          d_answer_support=float(np.mean(A["dsupp_ans"])),
                          d_alt_support=float(np.mean(A["dsupp_alt"])), n=len(A["before"]))
    return out

# ---------------- component attribution stability ----------------
def exp_component_stability(ctx, max_items=200, n_boot=30, top_k=8):
    """Bootstrap the component attribution: top-k overlap across resamples, rank correlation of the
    mean attribution vector, and late-layer / attention-vs-MLP mass fractions. Shows the component
    story is stable, not noisy top-5 archaeology."""
    import component_core as cc
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; Wu=ctx["Wu_np"]; mask=ctx["content_mask"]; L=ctx["L"]
    Xs=[]; aids=[]; cids=[]; names=None
    pool=[(it,fid(it["gold"][0])) for it in items]; pool=[(it,a) for it,a in pool if a is not None]
    for it,aid in pool:
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]): continue
        comp=_collect_components(ctx,p); cw=rw.content_winner(comp["z_final"],mask)["content_winner"]
        if cw==aid: continue
        Xs.append(comp["X"]); aids.append(aid); cids.append(cw); names=comp["names"]
        if len(Xs)>=max_items: break
    if len(Xs)<20: return dict(n=len(Xs), status="too_few")
    X=np.stack(Xs); n,K,d=X.shape; aids=np.array(aids); cids=np.array(cids)
    rng=np.random.default_rng(0)
    def attr_idx(idx):
        att=cc.attribute_margins(X[idx],Wu,aids[idx],cids[idx],use_loo=True)
        return att["mean_total"]
    full=attr_idx(np.arange(n)); full_top=set(np.argsort(full)[:top_k])
    overlaps=[]; rhos=[]
    for _ in range(n_boot):
        idx=rng.integers(0,n,n); v=attr_idx(idx)
        overlaps.append(len(set(np.argsort(v)[:top_k])&full_top)/top_k)
        rhos.append(mc.spearman(v,full))
    # mass fractions on the full attribution (toward-alternative = negative)
    neg=np.clip(-full,0,None)                                 # alternative-favoring magnitude per component
    is_attn=np.array([nm.startswith("attn.") for nm in names])
    is_mlp=np.array([nm.startswith("mlp.") for nm in names])
    layer_of=np.array([int(nm.split(".")[1]) if "." in nm else -1 for nm in names])
    late=layer_of>=int(0.6*L)
    tot=neg.sum()+1e-12
    return dict(n=n, n_boot=n_boot, top_k=top_k,
                mean_topk_overlap=float(np.mean(overlaps)), min_topk_overlap=float(np.min(overlaps)),
                mean_rank_corr=float(np.mean(rhos)),
                attn_mass_frac=float(neg[is_attn].sum()/tot), mlp_mass_frac=float(neg[is_mlp].sum()/tot),
                late_mass_frac=float(neg[late].sum()/tot),
                full_top_components=[names[i] for i in np.argsort(full)[:top_k]])

# ---------------- exp (exploratory): mean-residual / baseline ablation ----------------
def exp_mean_residual_ablation(ctx, freq, max_items=200, alphas=(0.5,1.0)):
    """EXPLORATORY: subtract a fraction of the mean residual (or remove only its projection onto the
    frequency direction) and measure selected-token frequency shift, margin shift, first-token recovery.
    Subtracting the full mean residual can break the representation broadly -- treat as exploratory."""
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; Wu=ctx["Wu_np"]; mask=ctx["content_mask"]
    H=[]; aids=[]; cids=[]
    pool=[(it,fid(it["gold"][0])) for it in items]; pool=[(it,a) for it,a in pool if a is not None]
    for it,aid in pool:
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]): continue
        R=_run_prompt(ctx,p); cw=rw.content_winner(R["z_final"],mask)["content_winner"]
        if cw==aid: continue
        H.append(R["h_final"]); aids.append(aid); cids.append(cw)
        if len(H)>=max_items: break
    if len(H)<10: return dict(n=len(H), status="too_few")
    H=np.array(H); hbar=H.mean(0)
    import baseline_core as bc
    r=bc.frequency_direction(Wu,freq); runit=r/ (np.linalg.norm(r)+1e-9)
    def measure(transform):
        rec=0; dfreq=[]; before=[]; after=[]
        for i in range(len(H)):
            h2=transform(H[i]); diff=Wu[aids[i]]-Wu[cids[i]]
            before.append(float(H[i]@diff)); after.append(float(h2@diff))
            new=int(np.argmax(np.where(mask,Wu@h2,-np.inf)))
            rec+=int(new==aids[i]); dfreq.append(float(freq[new]-freq[cids[i]]))
        import component_core as cc
        ms=cc.margin_shift(before,after)
        return dict(margin_shift=ms["mean_shift"], recovery=rec/len(H), mean_selected_freq_shift=float(np.mean(dfreq)))
    out=dict(n=len(H))
    for a in alphas:
        out[f"subtract_meanres_alpha{a}"]=measure(lambda h,a=a: h-a*hbar)
    out["remove_freqdir_projection"]=measure(lambda h: h-(h@runit)*runit)
    out["remove_meanres_projection"]=measure(lambda h: h-(h@(hbar/ (np.linalg.norm(hbar)+1e-9)))*(hbar/(np.linalg.norm(hbar)+1e-9)))
    return out


# ================= TUNED-LENS READ/WRITE =================

def _collect_layer_residuals(ctx, prompt, layers):
    """Capture the residual (hidden_states) at the last position for the given layers AND the final
    residual, for tuned-lens probe fitting. Returns dict layer-> (d,) and 'final'->(d,)."""
    m=ctx["m"]; tk=ctx["tk"]; L=ctx["L"]
    enc=tk(prompt,return_tensors="pt",truncation=True,max_length=256).to(ctx["device"])
    with torch.no_grad(): out=m(**enc)
    hs=out.hidden_states
    d={l: hs[l][0,-1,:].detach().float().cpu().numpy() for l in layers}
    d["final"]=hs[L][0,-1,:].detach().float().cpu().numpy()
    d["z_final"]=out.logits[0,-1,:].float().detach().cpu().numpy()
    return d

def exp_tuned_lens_readwrite(ctx, freq, n_fit=300, n_eval=400, l2=1.0):
    """Tuned-lens read/write dissociation. Fit per-layer translators (A_l h_l + b_l ~= h_final) on a
    HELD-OUT fit split, then recompute the read score with the tuned lens (everything else fixed: same
    failed items, gold first-token targets, same-relation decoys, hard-decoy criterion, final rank).
    Reports tuned hard-readable failure rate, fraction not-top-ranked, tuned read vs final-rank
    correlation, and the overlap between logit-lens-readable and tuned-lens-readable sets."""
    import tunedlens_core as tl
    items=ctx["items"]; fid=ctx["fid"]; QA=ctx["qa"]; Wu=ctx["Wu_np"]; mask=ctx["content_mask"]
    fn_t=ctx["fn"]; L=ctx["L"]; band=ctx["read_band"]
    def final_norm_np(hf):
        return fn_t(torch.tensor(hf,dtype=torch.float32,device=ctx["device"]).unsqueeze(0)).float().detach().cpu().numpy()[0]
    # --- fit split: collect (h_l, h_final) over band layers on the first n_fit items (any items) ---
    fit_items=[it for it in items[:n_fit]]
    Hbylayer={l:[] for l in band}; Hfinal=[]
    for it in fit_items:
        d=_collect_layer_residuals(ctx, QA.format(q=it["q"]), band)
        for l in band: Hbylayer[l].append(d[l])
        Hfinal.append(d["final"])
    Hbylayer={l:np.array(v) for l,v in Hbylayer.items()}; Hfinal=np.array(Hfinal)
    probes=tl.fit_all_layers(Hbylayer,Hfinal,band,l2=l2)
    r2=tl.reconstruction_quality(Hbylayer,Hfinal,probes,band)
    # --- eval split: DISJOINT items, recompute read with tuned lens ---
    eval_items=[it for it in items[n_fit:n_fit+n_eval]]
    rng=np.random.default_rng(0)
    tuned_read=[]; logit_read=[]; negwrite=[]; tuned_hardreadable=[]; logit_hardreadable=[]; nottop=[]; losing=[]
    n=0
    for it in eval_items:
        aid=fid(it["gold"][0])
        if aid is None: continue
        p=QA.format(q=it["q"])
        if _match(_greedy(ctx,p),it["gold"]): continue        # failed generations only
        d=_collect_layer_residuals(ctx,p,band)
        # tuned read = peak over band of tuned-lens logprob of the answer
        tlp=[]; llp=[]
        for l in band:
            A,b=probes[l]; lp_t=tl.tuned_logprob(d[l],A,b,Wu,final_norm_np); tlp.append(float(lp_t[aid]))
            hn=final_norm_np(d[l]); z=Wu@hn; z=z-z.max(); lp_l=z-np.log(np.exp(z).sum()); llp.append(float(lp_l[aid]))
        tr=max(tlp); lr=max(llp)
        # decoys: same-relation pool, tuned + logit reads, hard (max-decoy) criterion
        pool=[i for i in ctx["relpool"][it["rel"]] if i!=aid]
        dk=list(rng.choice(pool,min(8,len(pool)),replace=False)) if pool else []
        if dk:
            tdec=[]; ldec=[]
            for l in band:
                A,b=probes[l]; lp_t=tl.tuned_logprob(d[l],A,b,Wu,final_norm_np)
                hn=final_norm_np(d[l]); z=Wu@hn; z=z-z.max(); lp_l=z-np.log(np.exp(z).sum())
                tdec.append([float(lp_t[x]) for x in dk]); ldec.append([float(lp_l[x]) for x in dk])
            tdec=np.array(tdec).max(0); ldec=np.array(ldec).max(0)   # per-decoy peak over band
            t_hard=int(tr - tdec.max() > 0); l_hard=int(lr - ldec.max() > 0)
        else:
            t_hard=l_hard=0
        zf=d["z_final"]; rk=int((zf>zf[aid]).sum())
        tuned_read.append(tr); logit_read.append(lr); negwrite.append(-rk)
        tuned_hardreadable.append(t_hard); logit_hardreadable.append(l_hard); nottop.append(int(rk!=0)); losing.append(rk!=0)
        n+=1
    if n<10: return dict(n=n, status="too_few")
    tuned_read=np.array(tuned_read); logit_read=np.array(logit_read); negwrite=np.array(negwrite)
    th=np.array(tuned_hardreadable,bool); lh=np.array(logit_hardreadable,bool); nt=np.array(nottop)
    overlap=float((th&lh).sum()/max((th|lh).sum(),1))
    return dict(n=n, mean_recon_r2=float(np.mean(list(r2.values()))),
                tuned_hard_readable_rate=float(th.mean()), logit_hard_readable_rate=float(lh.mean()),
                tuned_nottop_among_readable=float(nt[th].mean()) if th.sum() else float("nan"),
                logit_nottop_among_readable=float(nt[lh].mean()) if lh.sum() else float("nan"),
                rho_tunedread_negwrite=mc.spearman(tuned_read,negwrite),
                rho_logitread_negwrite=mc.spearman(logit_read,negwrite),
                readable_set_overlap=overlap,
                n_tuned_readable=int(th.sum()), n_logit_readable=int(lh.sum()))

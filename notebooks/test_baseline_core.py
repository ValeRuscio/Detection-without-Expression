import numpy as np, baseline_core as bc
ok=True

# LOO mean correctness: loo[i] excludes row i
H=np.array([[1.,0],[3.,0],[5.,0],[7.,0]])
loo=bc.loo_mean(H)
# loo[0] = mean of rows 1,2,3 = (3+5+7)/3 = 5
ok=ok and abs(loo[0,0]-5.0)<1e-9 and abs(loo[1,0]-(1+5+7)/3)<1e-9
print(f"  [{'PASS' if abs(loo[0,0]-5.0)<1e-9 else 'FAIL'}] leave-one-out mean excludes own row (loo[0]={loo[0,0]:.3f})")

# EXACTNESS: baseline_margin + contextual_margin == final logit margin, to machine precision
rng=np.random.default_rng(0); V,d,n=500,32,80
Wu=rng.standard_normal((V,d)); H=rng.standard_normal((n,d))*2+rng.standard_normal(d)
aids=rng.integers(0,V,n); bids=rng.integers(0,V,n)
rows=bc.decompose_margin(Wu,H,aids,bids,use_loo=True)
maxres=max(abs(r["exact_residual"]) for r in rows)
ok=ok and maxres<1e-8
print(f"  [{'PASS' if maxres<1e-8 else 'FAIL'}] decomposition is EXACT (max residual {maxres:.2e})")

# planted baseline effect: make competitor have high baseline (via hbar aligned to competitor dirs)
# build H so the mean residual points toward a fixed 'frequent' set -> those tokens get high baseline
freqdir=Wu[10]/np.linalg.norm(Wu[10])
H2=rng.standard_normal((n,d))*0.5 + freqdir*4    # mean residual ~ along token-10 direction
aids2=rng.integers(0,V,n); bids2=np.full(n,10)   # competitor is always the 'baseline-favored' token 10
rows2=bc.decompose_margin(Wu,H2,aids2,bids2,use_loo=True)
agg2=bc.aggregate(rows2)
# competitor (token 10) has high baseline -> baseline_margin (ans - comp) should be NEGATIVE on average
ok=ok and agg2["mean_baseline_margin"]<0
print(f"  [{'PASS' if agg2['mean_baseline_margin']<0 else 'FAIL'}] planted baseline-favored competitor -> negative baseline margin ({agg2['mean_baseline_margin']:.2f})")

# regime labels
print(f"  baseline_limited example:", bc.regime_label(-5.0,-1.0))   # base more negative -> baseline_limited
print(f"  answer_support example:", bc.regime_label(+2.0,-4.0))     # base positive -> answer_support_limited
ok=ok and bc.regime_label(-5.0,-1.0)=="baseline_limited" and bc.regime_label(2.0,-4.0)=="answer_support_limited"

# alignment: baseline b_v should correlate with freq if hbar aligned to freq direction
freq=rng.uniform(-13,-3,V)
r=bc.frequency_direction(Wu,freq)
hbar=r*5   # mean residual exactly along frequency direction
al=bc.baseline_alignment(Wu,hbar,freq)
ok=ok and abs(al["cos_meanres_freqdir"]-1.0)<1e-6
print(f"  [{'PASS' if abs(al['cos_meanres_freqdir']-1.0)<1e-6 else 'FAIL'}] alignment: cos(meanres,freqdir)={al['cos_meanres_freqdir']:.3f} when hbar=r")
print(f"     corr(baseline, unembed freq proj)={al['corr_baseline_unembedfreqproj']:.3f} (should be ~1)")
ok=ok and al["corr_baseline_unembedfreqproj"]>0.99
print("ALL PASS" if ok else "SOME FAILED")

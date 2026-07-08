import numpy as np, tunedlens_core as tl
ok=True; rng=np.random.default_rng(0)

# translator recovers a known affine map
d,n=16,500
A_true=rng.standard_normal((d,d))*0.3+np.eye(d); b_true=rng.standard_normal(d)
Hl=rng.standard_normal((n,d))
Hf=Hl@A_true.T+b_true+rng.standard_normal((n,d))*0.01
A,b=tl.fit_translator(Hl,Hf,l2=0.1)
err=np.max(np.abs(A-A_true))
ok=ok and err<0.1
print(f"  [{'PASS' if err<0.1 else 'FAIL'}] translator recovers affine map (max coef err {err:.3f})")

# reconstruction R^2 high when relationship is affine
probes={5:(A,b)}; r2=tl.reconstruction_quality({5:Hl},Hf,probes,[5])
ok=ok and r2[5]>0.98
print(f"  [{'PASS' if r2[5]>0.98 else 'FAIL'}] reconstruction R2={r2[5]:.3f}")

# tuned_logprob returns a valid log-prob distribution
V=200; Wu=rng.standard_normal((V,d)); fn=lambda h: h/ (np.sqrt((h**2).mean())+1e-6)
lp=tl.tuned_logprob(Hl[0],A,b,Wu,fn)
ok=ok and abs(np.exp(lp).sum()-1.0)<1e-6 and lp.shape==(V,)
print(f"  [{'PASS' if abs(np.exp(lp).sum()-1)<1e-6 else 'FAIL'}] tuned_logprob is a valid distribution (sum={np.exp(lp).sum():.4f})")

# held-out generalization: fit on train, R2 still high on test (no leakage needed for the math)
Hl_tr,Hl_te=Hl[:400],Hl[400:]; Hf_tr,Hf_te=Hf[:400],Hf[400:]
A2,b2=tl.fit_translator(Hl_tr,Hf_tr,l2=0.1)
r2_te=tl.reconstruction_quality({0:Hl_te},Hf_te,{0:(A2,b2)},[0])[0]
ok=ok and r2_te>0.95
print(f"  [{'PASS' if r2_te>0.95 else 'FAIL'}] held-out R2={r2_te:.3f} (generalizes)")
print("ALL PASS" if ok else "SOME FAILED")

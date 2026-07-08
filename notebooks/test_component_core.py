import numpy as np, component_core as cc
ok=True

# LOO stack mean excludes own item
X=np.zeros((4,2,3)); X[:,0,0]=[1,3,5,7]
loo=cc.loo_mean_stack(X)
ok=ok and abs(loo[0,0,0]-5.0)<1e-9   # mean of 3,5,7
print(f"  [{'PASS' if abs(loo[0,0,0]-5)<1e-9 else 'FAIL'}] LOO stack mean excludes own item")

# attribution sums to final margin: sum_k total_{i,k} == <h_i, W_U[a]-W_U[b]>
rng=np.random.default_rng(0); n,K,d,V=60,8,16,300
X=rng.standard_normal((n,K,d)); Wu=rng.standard_normal((V,d))
aids=rng.integers(0,V,n); bids=rng.integers(0,V,n)
att=cc.attribute_margins(X,Wu,aids,bids,use_loo=True)
H=X.sum(1)  # (n,d)
final=np.array([ (Wu[aids[i]]-Wu[bids[i]])@H[i] for i in range(n)])
summed=att["total"].sum(1)
ok=ok and np.max(np.abs(summed-final))<1e-8
print(f"  [{'PASS' if np.max(np.abs(summed-final))<1e-8 else 'FAIL'}] component totals sum to final margin (max err {np.max(np.abs(summed-final)):.1e})")

# baseline + contextual == total per component
ok=ok and np.max(np.abs(att["baseline"]+att["contextual"]-att["total"]))<1e-9
print(f"  [{'PASS' if np.max(np.abs(att['baseline']+att['contextual']-att['total']))<1e-9 else 'FAIL'}] baseline+contextual==total per component")

# ranking: plant one component that strongly favors the competitor
X2=rng.standard_normal((n,K,d))*0.2
# make component 3 point along -(Wu[a]-Wu[b]) per item -> strongly negative total for comp 3
for i in range(n):
    diff=Wu[aids[i]]-Wu[bids[i]]; X2[i,3]= -diff/ (np.linalg.norm(diff)+1e-9)*5
att2=cc.attribute_margins(X2,Wu,aids,bids,use_loo=True)
names=[f"comp{c}" for c in range(K)]
top=cc.rank_components(att2["mean_total"],names,k=3,most_negative=True)
ok=ok and top[0][0]=="comp3"
print(f"  [{'PASS' if top[0][0]=='comp3' else 'FAIL'}] ranking finds planted alternative-favoring component (top={top[0]})")

# margin shift
ms=cc.margin_shift([-2,-1,0,1],[0,1,2,3])
ok=ok and abs(ms["mean_shift"]-2.0)<1e-9 and ms["frac_improved"]==1.0
print(f"  [{'PASS' if abs(ms['mean_shift']-2)<1e-9 else 'FAIL'}] margin shift (mean_shift={ms['mean_shift']})")
print("ALL PASS" if ok else "SOME FAILED")

"""Diagnose the non-monotone shot-limited kernel (thesis Table 5.17.1).
Same data, split, bandwidth (0.15) and alpha grid as experiments_realistic.shot_limited_kernel.
Arms: (a) alpha re-selected by grouped CV per draw [as in thesis]; (b) alpha fixed at the
exact-Gram CV choice; (c) alpha fixed, no PSD projection."""
import sys, os, time, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from bench import rmse
from experiments import qk_states_n, quantum_gram
from experiments_realistic import split_random, PARAM_FEATS
A = pd.read_csv("ds_real_A.csv"); trA, teA = split_random(A)
sc = StandardScaler().fit(A.loc[trA, PARAM_FEATS].to_numpy())
Xtr = sc.transform(A.loc[trA, PARAM_FEATS].to_numpy()); Xte = sc.transform(A.loc[teA, PARAM_FEATS].to_numpy())
ytr, yte = A.loc[trA,"ideal"].to_numpy(), A.loc[teA,"ideal"].to_numpy()
gtr = A.loc[trA,"circuit_id"].to_numpy()
Ptr, Pte = qk_states_n(Xtr, 0.15), qk_states_n(Xte, 0.15)
K, Kt = quantum_gram(Ptr), quantum_gram(Pte, Ptr)
grid = {"alpha": np.logspace(-6, 1, 12)}
def cvfit(Ktr):
    return GridSearchCV(KernelRidge(kernel="precomputed"), grid, cv=GroupKFold(n_splits=4), n_jobs=1).fit(Ktr, ytr, groups=gtr)
g0 = cvfit(K); a0 = g0.best_params_["alpha"]
rows=[{"shots":0,"draw":0,"arm":"exact","alpha":a0,"rmse":rmse(yte,np.clip(g0.predict(Kt),-1.5,1.5)),"min_eig_raw":float(np.linalg.eigvalsh(K)[0])}]
print("exact", rows[-1], flush=True)
rng = np.random.default_rng(2026)
iu = np.triu_indices(len(trA))
for M in [100, 1000, 10000, 100000]:
    for d in range(5):
        t=time.time()
        Kr = np.zeros_like(K); Kr[iu] = rng.binomial(M, np.clip(K[iu],0,1))/M
        Kr = Kr + Kr.T - np.diag(np.diag(Kr))
        Kte = rng.binomial(M, np.clip(Kt,0,1))/M
        w,V = np.linalg.eigh(Kr); Kp = (V*np.clip(w,0,None))@V.T
        nneg = int((w<0).sum()); mineig=float(w[0])
        g = cvfit(Kp)
        rows.append({"shots":M,"draw":d,"arm":"cv_alpha_psd","alpha":g.best_params_["alpha"],"rmse":rmse(yte,np.clip(g.predict(Kte),-1.5,1.5)),"min_eig_raw":mineig,"n_neg_eig":nneg})
        m = KernelRidge(kernel="precomputed", alpha=a0).fit(Kp, ytr)
        rows.append({"shots":M,"draw":d,"arm":"fixed_alpha_psd","alpha":a0,"rmse":rmse(yte,np.clip(m.predict(Kte),-1.5,1.5)),"min_eig_raw":mineig,"n_neg_eig":nneg})
        try:
            m2 = KernelRidge(kernel="precomputed", alpha=a0).fit(Kr, ytr)
            r2 = rmse(yte,np.clip(m2.predict(Kte),-1.5,1.5))
        except Exception as e: r2=np.nan
        rows.append({"shots":M,"draw":d,"arm":"fixed_alpha_noproj","alpha":a0,"rmse":r2,"min_eig_raw":mineig,"n_neg_eig":nneg})
        print(M,d,[ (r["arm"],round(r["rmse"],4),r["alpha"]) for r in rows[-3:]], nneg, round(mineig,3), f"{time.time()-t:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv("diag_shotkernel.csv", index=False)

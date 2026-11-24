import sys, os, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, os.getcwd())
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
ytr, yte = A.loc[trA,"ideal"].to_numpy(), A.loc[teA,"ideal"].to_numpy(); gtr = A.loc[trA,"circuit_id"].to_numpy()
K, Kt = quantum_gram(qk_states_n(Xtr,0.15)), quantum_gram(qk_states_n(Xte,0.15), qk_states_n(Xtr,0.15))
fine = np.logspace(-3, 1, 25)
rng = np.random.default_rng(2026); iu = np.triu_indices(len(trA)); rows=[]
for M in [100, 1000, 10000, 100000]:
    for d in range(5):
        Kr = np.zeros_like(K); Kr[iu] = rng.binomial(M, np.clip(K[iu],0,1))/M; Kr = Kr + Kr.T - np.diag(np.diag(Kr))
        Kte = rng.binomial(M, np.clip(Kt,0,1))/M
        w,V = np.linalg.eigh(Kr); Kp = (V*np.clip(w,0,None))@V.T
        g = GridSearchCV(KernelRidge(kernel="precomputed"), {"alpha": fine}, cv=GroupKFold(4), n_jobs=1).fit(Kp, ytr, groups=gtr)
        r_cv = rmse(yte, np.clip(g.predict(Kte),-1.5,1.5))
        test_curve = [rmse(yte, np.clip(KernelRidge(kernel="precomputed", alpha=a).fit(Kp,ytr).predict(Kte),-1.5,1.5)) for a in fine]
        j = int(np.argmin(test_curve))
        rows.append({"shots":M,"draw":d,"cv_alpha":g.best_params_["alpha"],"rmse_cv_fine":r_cv,"oracle_alpha":fine[j],"rmse_oracle":test_curve[j],"min_eig":float(w[0])})
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv("diag_finegrid.csv", index=False)

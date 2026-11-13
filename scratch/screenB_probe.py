"""SCREEN B, step 2 -- is the ENVIRONMENT recoverable from a labelled context?

Same methodology that correctly predicted the geometry failure: regress the
latent from features + KNOWN source positions, train and test on DISJOINT
environments. Chance = predicting the training mean.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

d = np.load("scratch/screenB_env.npz")
F, SRC, LAT, ENV = d["feats"], d["src"], d["lat"], d["env"]
tau, S = F.shape[1], F.shape[2]
print(f"  {F.shape[0]} rows | tau={tau} sensors={S} | {len(set(ENV.tolist()))} environments")

# same compact delay features the localiser uses
MPT = 1500.0 / (tau * float(d["df"])); ROLL = 50
A = np.abs(F); pk = A.max(1)
first = np.argmax(A >= 0.30 * pk[:, None, :], axis=1).astype(np.float32)
peak = A.argmax(1).astype(np.float32)
e = A ** 2; tot = e.sum(1) + 1e-30; taps = np.arange(tau, dtype=np.float32)
mt = (e * taps[None, :, None]).sum(1) / tot
spread = np.sqrt(np.maximum((e * (taps[None, :, None] - mt[:, None, :]) ** 2).sum(1) / tot, 0))
X = np.c_[(first - ROLL) * MPT / 1000.0, (peak - ROLL) * MPT / 1000.0,
          np.log10(pk + 1e-12) / 10.0, spread * MPT / 1000.0]
print(f"  delay-feature dim = {X.shape[1]}")

rng = np.random.default_rng(0); envs = np.array(sorted(set(ENV.tolist())))
tr_e = set(rng.permutation(envs)[:int(0.7 * len(envs))].tolist())
trm = np.array([v in tr_e for v in ENV]); tem = ~trm
def z(M): return (M - M[trm].mean(0)) / (M[trm].std(0) + 1e-9)
XX = np.c_[z(X), z(SRC)]
names = ["sound speed c [m/s]", "depth h0 [m]", "bottom speed c2 [m/s]"]
print(f"\n  train {len(tr_e)} envs / test {len(envs)-len(tr_e)} envs (DISJOINT)")
print(f"  {'latent':<22}{'ridge':>9}{'mlp':>9}{'per-env mlp':>13}{'chance':>9}")
for i, nm in enumerate(names):
    y = LAT[:, i]
    ch = np.abs(y[tem] - y[trm].mean()).mean()
    r = Ridge(1.0).fit(XX[trm], y[trm]); pr = np.abs(r.predict(XX[tem]) - y[tem]).mean()
    m = MLPRegressor(hidden_layer_sizes=(256,128), max_iter=220,
                     random_state=0).fit(XX[trm], y[trm])
    pm_pt = m.predict(XX[tem]); pm = np.abs(pm_pt - y[tem]).mean()
    # a CONTEXT SET pools many points -> average the prediction per environment
    pe = []
    for v in envs[[q not in tr_e for q in envs]]:
        msk = ENV[tem] == v
        if msk.any(): pe.append(abs(pm_pt[msk].mean() - y[tem][msk][0]))
    print(f"  {nm:<22}{pr:>9.1f}{pm:>9.1f}{np.mean(pe):>13.1f}{ch:>9.1f}")

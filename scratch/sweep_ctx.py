import os, sys, glob
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
M = {}
for tag, root in [("FIX","data/pilot_broadband/hermite_shared"),
                  ("CONTROL","data/pilot_legacy/hermite_shared")]:
    FE, SRC, GID, POSL = [], [], [], []
    for gi, sd in enumerate(sorted(glob.glob(f"{root}/position_set_*"))):
        b = f"{sd}/channel_option_0.1/random"
        F = np.load(f"{b}/filtered_data/filtered_data.npy")
        pos = np.load(glob.glob(f"{b}/channel_info/sensor_positions_*.npy")[0])
        tr = np.load(f"{b}/trajectory/trajectories.npy")
        tau, ppt, ntr, ns = F.shape
        FE.append(F.transpose(2,1,0,3).reshape(-1,tau,ns))
        SRC.append(tr.transpose(1,2,0)[:,:,:2].reshape(-1,2))
        GID.append(np.full(ntr*ppt, gi)); POSL.append(pos[:2].T)
    M[tag] = (np.concatenate(FE), np.concatenate(SRC), np.concatenate(GID), POSL, tau, ns)

print("  n_ctx |   FIX (broadband+timing) |  CONTROL (legacy tone)")
for C in [4,5,8,10,20,30,50]:
    row = []
    for tag in ("FIX","CONTROL"):
        FE,SRC,GID,POSL,tau,ns = M[tag]
        MPT = 1500.0/tau; ROLL=50
        rng = np.random.default_rng(0); errs=[]
        for gi in range(len(POSL)):
            m = GID==gi; Fg,Sg = FE[m],SRC[m]
            sel = rng.choice(len(Fg), size=min(C,len(Fg)), replace=False)
            env = np.abs(Fg[sel])
            thr = 0.30*env.max(axis=1, keepdims=True)
            r_est = (np.argmax(env>=thr,axis=1)-ROLL)*MPT
            for s in range(ns):
                P,r = Sg[sel], r_est[:,s]
                A = np.c_[-2*P, np.ones(len(P))]
                sol,*_ = np.linalg.lstsq(A, r**2-(P**2).sum(1), rcond=None)
                errs.append(np.linalg.norm(sol[:2]-POSL[gi][s]))
        row.append(np.median(errs))
    print(f"  {C:5d} |   {row[0]:10.1f} m            |  {row[1]:10.1f} m")

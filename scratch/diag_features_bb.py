"""ORACLE fails WITH true positions, and cannot even fit (train 359 vs chance 481)
while multilateration on the same features reaches 3.3 m. So the information is
PRESENT but not ACCESSIBLE to the network. Prime suspect: dynamic range."""
import os, glob
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
b = "data/random_task_bb/hermite_distinct/position_set_00/channel_option_0.1/random"
F = np.load(f"{b}/filtered_data/filtered_data.npy", mmap_mode='r')
tr = np.load(f"{b}/trajectory/trajectories.npy")
pos = np.load(glob.glob(f"{b}/channel_info/sensor_positions_*.npy")[0])
tau, ppt, ntr, ns = F.shape
W = np.asarray(F[:, :, :20, :]).transpose(2,1,3,0)     # (20,ppt,ns,tau)
src = tr.transpose(1,2,0)[:20,:,:2]
d = np.linalg.norm(src[:,:,None,:] - pos[:2].T[None,None,:,:], axis=-1)  # (20,ppt,ns)
pk = np.abs(W).max(axis=-1)                             # per-record peak
R, P = d.ravel(), pk.ravel()
print(f"PER-RECORD PEAK AMPLITUDE (what the encoder must normalise)")
for q in [0,1,25,50,75,99,100]:
    print(f"   p{q:<3d} = {np.percentile(P,q):.3e}")
print(f"   dynamic range max/min = {P.max()/max(P.min(),1e-30):.3e}  "
      f"({np.log10(P.max()/max(P.min(),1e-30)):.1f} orders of magnitude)")
print(f"   corr(log peak, log range) = {np.corrcoef(np.log(P+1e-30), np.log(R+1e-9))[0,1]:+.3f}")
gstd = float(np.asarray(F[:,:,:20,:]).std())
print(f"\nGLOBAL std (what norm_acoustic=standardize divides by) = {gstd:.3e}")
print(f"   after global standardize, per-record peak becomes:")
for q in [1,50,99]:
    print(f"     p{q:<3d} = {np.percentile(P,q)/gstd:9.3f}")
near = P[R < np.percentile(R,10)]; far = P[R > np.percentile(R,90)]
print(f"   nearest-10% records -> peak/std ~ {np.median(near)/gstd:9.3f}")
print(f"   farthest-10% records -> peak/std ~ {np.median(far)/gstd:9.3f}")
print(f"   ratio near/far = {np.median(near)/max(np.median(far),1e-30):.1f}x")
frac = (np.abs(W) > 0.1*np.abs(W).max(axis=-1, keepdims=True)).mean()
print(f"\nSPARSITY: fraction of taps above 10% of their record peak = {frac*100:.2f}%")
print(f"   -> the cue is the INDEX of a spike, not a smooth spectrum")

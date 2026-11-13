"""Is the TOA still in filtered_data, just not visible without pulse compression?
Rebuild the transmitted chirp, matched-filter the stored record, re-check TOA."""
import os, sys, glob
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
b = "data/pilot_broadband/hermite_shared/position_set_00/channel_option_0.1/random"
F = np.load(f"{b}/filtered_data/filtered_data.npy")
h = np.load(glob.glob(f"{b}/channel_info/channel_h_*.npy")[0])
tr = np.load(f"{b}/trajectory/trajectories.npy")
pos = np.load(glob.glob(f"{b}/channel_info/sensor_positions_*.npy")[0])
tau, ppt, ntr, ns = F.shape
src = tr.transpose(1,2,0)[:,:,:2]
d = np.linalg.norm(src[:,:,None,:] - pos[:2].T[None,None,:,:], axis=-1)   # (ntr,ppt,ns)
M_PER_TAP = 1500.0/tau

# transmitted chirp, exactly as acoustic_data_generator builds it
n = 1024; f0, f1 = 0.02, 0.45
kk = np.arange(n)
x = np.cos(2*np.pi*(f0*kk + 0.5*(f1-f0)/(n-1)*kk**2)); x /= np.sqrt((x**2).sum())

W = F.transpose(2,1,3,0).reshape(-1, tau)                 # (N,tau) records
D = d.reshape(-1)
sel = np.random.default_rng(0).choice(len(W), 3000, replace=False)
def report(P, name):
    r = np.corrcoef(P, D[sel])[0,1]
    q = np.polyfit(P, D[sel], 1); e = np.abs(np.polyval(q,P)-D[sel])
    print(f"  {name:34s} corr={r:+.4f}  median|range err|={np.median(e):7.1f} m")

report(np.abs(W[sel]).argmax(1).astype(float), "raw record, envelope peak")
# matched filter: correlate each record with the chirp
mf = np.array([np.correlate(W[i], x, mode='full') for i in sel])
off = len(x) - 1
report(np.abs(mf[:, off:off+tau]).argmax(1).astype(float), "matched-filtered, envelope peak")
report(np.abs(mf).argmax(1).astype(float) - off, "matched-filtered, full-support peak")
# reference: the impulse response itself
env_h = np.abs(h).transpose(2,1,3,0).reshape(-1, tau)[sel]
report(env_h.argmax(1).astype(float), "impulse response h (reference)")
print(f"\n  ranges {D.min():.0f}-{D.max():.0f} m ; {M_PER_TAP:.2f} m per tap")

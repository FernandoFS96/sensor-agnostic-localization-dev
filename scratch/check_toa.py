"""Does absolute_delay actually put time-of-flight into the impulse response?
Peak tap should track true range with slope c/(Lf*df) metres per tap."""
import os, sys, glob
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
base = sys.argv[1]
h = np.load(glob.glob(f"{base}/channel_info/channel_h_*.npy")[0])
tr = np.load(f"{base}/trajectory/trajectories.npy")
pos = np.load(glob.glob(f"{base}/channel_info/sensor_positions_*.npy")[0])
tau, ppt, ntr, ns = h.shape
env = np.abs(h).transpose(2,1,3,0)                       # (ntr,ppt,ns,tau)
src = tr.transpose(1,2,0)[:,:,:2]                        # (ntr,ppt,2)
d = np.linalg.norm(src[:,:,None,:] - pos[:2].T[None,None,:,:], axis=-1)
pk = env.argmax(axis=-1).astype(float)
D, P = d.ravel(), pk.ravel()
r = np.corrcoef(P, D)[0,1]
q = np.polyfit(P, D, 1)
err = np.abs(np.polyval(q, P) - D)
print(f"  h {h.shape}   ranges {D.min():.0f}-{D.max():.0f} m")
print(f"  corr(peak tap, true range) = {r:+.4f}")
print(f"  fitted slope = {q[0]:.2f} m/tap   (expected c/(Lf*df) = {1500/(tau*1.0):.2f})"
      if len(sys.argv)<3 else "")
print(f"  median |range error| from peak tap = {np.median(err):.2f} m   "
      f"(90th pct {np.percentile(err,90):.1f} m)")

"""The simulator computes the FULL broadband channel over 10-20 kHz (201 bins)
and SAVES it (channel_h_*.npy). The single-tone probe collapses it to one
frequency. Question: does the broadband impulse response carry the time-of-
arrival -> range information that geometry inference needs?"""
import os, glob, numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
base = "data/random_task/hermite_shared/position_set_00/channel_option_0.1/random"
h = np.load(glob.glob(f"{base}/channel_info/channel_h_*.npy")[0], mmap_mode='r')
print(f"impulse response tensor: {h.shape} {h.dtype}   (tau, ppt, n_traj, n_sensors)")
tau, ppt, ntr, ns = h.shape
tr  = np.load(f"{base}/trajectory/trajectories.npy")
pos = np.load(glob.glob(f"{base}/channel_info/sensor_positions_*.npy")[0])

NT = 30
H = np.asarray(h[:, :, :NT, :])                       # (tau, ppt, NT, ns)
env = np.abs(H).transpose(2, 1, 3, 0)                 # (NT, ppt, ns, tau)
src = tr.transpose(1, 2, 0)[:NT, :, :2]               # (NT, ppt, 2)
d = np.linalg.norm(src[:, :, None, :] - pos[:2].T[None, None, :, :], axis=-1)  # (NT,ppt,ns)

# time-of-arrival proxy: first tap above a fraction of the record peak
thr = 0.25 * env.max(axis=-1, keepdims=True)
toa = np.argmax(env >= thr, axis=-1).astype(float)    # (NT,ppt,ns)
pk  = env.argmax(axis=-1).astype(float)

flat_d = d.ravel()
for name, v in [("first-arrival tap", toa.ravel()), ("peak tap", pk.ravel())]:
    r = np.corrcoef(v, flat_d)[0, 1]
    q = np.polyfit(v, flat_d, 1); err = np.abs(np.polyval(q, v) - flat_d)
    print(f"  corr({name:18s}, true range) = {r:+.3f}   "
          f"median |range err| from linear fit = {np.median(err):6.1f} m")

# compare against the single-tone amplitude cue measured earlier
A = np.load(f"{base}/filtered_data/filtered_data.npy", mmap_mode='r')
rms = np.sqrt((np.asarray(A[:, :, :NT, :])**2).mean(0)).transpose(1, 2, 0)   # (NT,ppt,ns)
r_amp = np.corrcoef(np.log(rms.ravel()+1e-12), np.log(flat_d+1e-9))[0, 1]
print(f"\n  (single-tone amplitude cue, for comparison: corr = {r_amp:+.3f})")
print(f"  ranges span {flat_d.min():.0f}-{flat_d.max():.0f} m")

# effective rank of the broadband envelope
E = env.reshape(-1, tau); Ec = E - E.mean(0, keepdims=True)
sv = np.linalg.svd(Ec[np.random.default_rng(0).choice(len(Ec), 4000, replace=False)],
                   compute_uv=False)
ev = sv**2/(sv**2).sum()
print(f"  broadband envelope effective rank: {int(np.searchsorted(np.cumsum(ev),0.90)+1)} "
      f"components for 90% variance (single-tone record was 2)")

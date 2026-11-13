"""WHY is geometry not identifiable? Hypothesis: the transmit waveform is a single
tone, so each sensor's 201-sample record carries essentially ONE complex number
(amplitude, phase at one frequency) -- and phase at 10-20 kHz is aliased far
below the 100 m ranges here, leaving multipath-corrupted AMPLITUDE as the only
geometry cue. Test it: spectral concentration + effective rank."""
import os, glob, numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
f = glob.glob("data/random_task/hermite_shared/position_set_00/channel_option_0.1/random/filtered_data/filtered_data.npy")[0]
A = np.load(f)                                  # (tau=201, ppt, n_traj, n_sensors)
tau, ppt, ntr, ns = A.shape
W = A[:, :, :20, :].transpose(1, 2, 3, 0).reshape(-1, tau)   # (N, 201) per-sensor records
print(f"per-(sensor,point) record length tau={tau}; probing N={W.shape[0]} records\n")

# --- 1. spectral concentration -------------------------------------------------
S = np.abs(np.fft.rfft(W, axis=1))**2
S = S / (S.sum(1, keepdims=True) + 1e-30)
top1 = np.sort(S, axis=1)[:, -1].mean()
top3 = np.sort(S, axis=1)[:, -3:].sum(1).mean()
peak = np.bincount(S.argmax(1), minlength=S.shape[1]).argmax()
print(f"1. SPECTRAL CONCENTRATION")
print(f"   mean energy fraction in the single largest FFT bin : {top1*100:5.1f}%")
print(f"   mean energy fraction in the top-3 bins             : {top3*100:5.1f}%")
print(f"   modal peak bin = {peak} of {S.shape[1]}  (a broadband probe would spread energy)")

# --- 2. effective rank ---------------------------------------------------------
Wc = W - W.mean(0, keepdims=True)
sv = np.linalg.svd(Wc, compute_uv=False)
ev = sv**2 / (sv**2).sum()
k90 = int(np.searchsorted(np.cumsum(ev), 0.90) + 1)
k99 = int(np.searchsorted(np.cumsum(ev), 0.99) + 1)
print(f"\n2. EFFECTIVE RANK of the 201-dim record")
print(f"   components for 90% variance: {k90} / {tau}")
print(f"   components for 99% variance: {k99} / {tau}")
print(f"   top-2 singular values carry {ev[:2].sum()*100:.1f}% of variance"
      f"  -> ~(amplitude, phase) at ONE frequency" if ev[:2].sum() > 0.5 else "")

# --- 3. is amplitude a usable range cue? --------------------------------------
tr = np.load("data/random_task/hermite_shared/position_set_00/channel_option_0.1/random/trajectory/trajectories.npy")
pos = np.load(glob.glob("data/random_task/hermite_shared/position_set_00/channel_option_0.1/random/channel_info/sensor_positions_*.npy")[0])
src = tr.transpose(1,2,0)[:20,:,:2].reshape(-1,2)                 # (N,2)
rms = np.sqrt((A[:,:,:20,:]**2).mean(0)).transpose(1,2,0).reshape(-1,ns)  # (N,ns)
d = np.linalg.norm(src[:,None,:] - pos[:2].T[None,:,:], axis=-1)  # (N,ns) true range
lr, ld = np.log(rms.ravel()+1e-12), np.log(d.ravel()+1e-9)
r = np.corrcoef(lr, ld)[0,1]
print(f"\n3. AMPLITUDE-vs-RANGE (the only surviving geometry cue)")
print(f"   corr(log RMS amplitude, log true range) = {r:+.3f}  over {len(lr)} (sensor,point) pairs")
print(f"   -> {'usable' if abs(r)>0.7 else 'WEAK: multipath destroys the range law'}")
# scatter of range implied by amplitude
q = np.polyfit(lr, ld, 1); pred = np.exp(np.polyval(q, lr))
print(f"   median |range error| from a fitted amplitude->range law: "
      f"{np.median(np.abs(pred - d.ravel())):.1f} m  (ranges span {d.min():.0f}-{d.max():.0f} m)")

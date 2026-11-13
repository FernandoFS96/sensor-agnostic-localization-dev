"""Proposed receiver chain: FULL convolution -> matched filter -> window at the
delay origin.  z = correlate(conv(x,h), x, 'full')[n-1 : n-1+tau]  should be the
pulse-compressed impulse response, with tap k == delay k."""
import os, glob
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
b = "data/pilot_broadband/hermite_shared/position_set_00/channel_option_0.1/random"
h = np.load(glob.glob(f"{b}/channel_info/channel_h_*.npy")[0])
tr = np.load(f"{b}/trajectory/trajectories.npy")
pos = np.load(glob.glob(f"{b}/channel_info/sensor_positions_*.npy")[0])
tau, ppt, ntr, ns = h.shape
d = np.linalg.norm(tr.transpose(1,2,0)[:,:,None,:2] - pos[:2].T[None,None,:,:], axis=-1)
n = 1024; f0, f1 = 0.02, 0.45; kk = np.arange(n)
x = np.cos(2*np.pi*(f0*kk + 0.5*(f1-f0)/(n-1)*kk**2)); x /= np.sqrt((x**2).sum())
hr = np.roll(h, 50, axis=0)          # filter() applies this roll
rng = np.random.default_rng(0)
idx = [(i,p,s) for i in range(ntr) for p in range(ppt) for s in range(ns)]
sel = [idx[i] for i in rng.choice(len(idx), 1500, replace=False)]
peaks, rngs = [], []
for (i,p,s) in sel:
    hh = np.real(hr[:, p, i, s])
    y = np.convolve(x, hh, 'full')                      # FULL, not 'valid'[0:tau]
    nz = rng.standard_normal(len(y))
    sp, npow = (y**2).sum(), (nz**2).sum()
    y = y + nz*np.sqrt(10**(-10/10)*sp/npow)            # SNR 10 dB
    z = np.correlate(y, x, 'full')[n-1 : n-1+tau]       # pulse compression, delay-aligned
    peaks.append(np.abs(z).argmax()); rngs.append(d[i,p,s])
peaks = np.array(peaks, float); rngs = np.array(rngs)
r = np.corrcoef(peaks, rngs)[0,1]
q = np.polyfit(peaks, rngs, 1); e = np.abs(np.polyval(q,peaks)-rngs)
print(f"  matched-filter + delay-aligned window (SNR 10 dB):")
print(f"    corr(peak tap, true range) = {r:+.4f}")
print(f"    slope {q[0]:.2f} m/tap (expect {1500/tau:.2f}), intercept {q[1]:.0f} m (the roll-50 offset)")
print(f"    median |range err| = {np.median(e):.2f} m | 90th pct {np.percentile(e,90):.1f} m")

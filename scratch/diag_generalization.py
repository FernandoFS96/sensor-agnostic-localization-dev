"""ORACLE now FITS (train 57 m) but does not GENERALISE (val 395 m) -- it is
memorising the 70 training layouts instead of learning trilateration.
Hypothesis: FourierPositionEncoding(max_wavelength=1000 m) ALIASES on a field
that is now ~1500 m (sensors) / ~2600 m (sources) wide, so the model cannot form
the range<->position relation and falls back to per-layout memorisation.

Test outside the NP entirely: plain MLP on [10 ranges, sensor positions] -> xy,
trained on train geometries, evaluated on HELD-OUT geometries, under three
position encodings."""
import os, pickle, math
import numpy as np
from sklearn.neural_network import MLPRegressor
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = "data/random_task_bb/hermite_distinct/processed_delay/geometry_split"
def load(f, stride):
    s = pickle.load(open(f"{D}/{f}", "rb"))
    R, S, Y = [], [], []
    for t in s[::stride]:
        X = np.asarray(t["X"]); sp = np.asarray(t["sensor_pos"])[:, :2]
        R.append(X[:, :10]); Y.append(np.asarray(t["y"])[:, :2])
        S.append(np.repeat(sp.reshape(1, -1), X.shape[0], axis=0))
    return np.concatenate(R), np.concatenate(S), np.concatenate(Y)
Rtr, Str, Ytr = load("train_data.pkl", 40)
Rte, Ste, Yte = load("val_data.pkl", 40)
print(f"train rows {len(Rtr)} (70 layouts) | held-out rows {len(Rte)} (14 layouts)")
ch = np.linalg.norm(Yte - Ytr.mean(0), axis=1).mean()

def fourier(P, n_bands=8, minw=10.0, maxw=1000.0):
    w = np.logspace(math.log10(minw), math.log10(maxw), n_bands)
    f = 2*np.pi/w
    p = P.reshape(len(P), -1, 1) * f
    return np.concatenate([np.sin(p), np.cos(p)], -1).reshape(len(P), -1)

for name, enc in [("raw positions /1000", lambda P: P/1000.0),
                  ("Fourier max_wl=1000 m  (CURRENT)", lambda P: fourier(P, maxw=1000.0)),
                  ("Fourier max_wl=6000 m  (proposed)", lambda P: fourier(P, maxw=6000.0))]:
    Xtr = np.c_[Rtr, enc(Str)]; Xte = np.c_[Rte, enc(Ste)]
    m = MLPRegressor(hidden_layer_sizes=(256,256,128), max_iter=60,
                     random_state=0, early_stopping=False)
    m.fit(Xtr, Ytr/1000.0)
    e = np.linalg.norm(m.predict(Xte)*1000.0 - Yte, axis=1)
    tr = np.linalg.norm(m.predict(Xtr)*1000.0 - Ytr, axis=1)
    print(f"  {name:36s} train {tr.mean():6.1f} m | HELD-OUT {e.mean():6.1f} m")
print(f"  {'chance':36s} {'':13s}| {ch:11.1f} m")
print(f"  {'analytic trilateration':36s} {'':13s}| {14.0:11.1f} m")

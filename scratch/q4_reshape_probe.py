"""Q4: empirically verify the feature axis ordering end-to-end.
Marks ONE sensor's entire channel with a sentinel in a REAL filtered_data tensor,
runs it through the ACTUAL processor reshape (reshape_input_data) and the ACTUAL
encoder reshape (SpatialEncoder: view(...,tau,n_sensors).transpose(-1,-2)), and
checks the sentinel lands in the right sensor slot. Also shows the WRONG reshape
scrambles it (the silent-failure mode)."""
import os,sys,glob,numpy as np, torch as t
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,"data")
import process_data as P   # the real processor (reshape_input_data)

# --- load a REAL processed sample too, to confirm shapes ---
import pickle
pk=pickle.load(open("data/random_task/hermite_distinct/processed/geometry_split/val_data.pkl","rb"))
s0=pk[0]
print("processed sample keys:", list(s0.keys()))
for k in ("X","y","sensor_pos"):
    print(f"  {k}: shape={np.asarray(s0[k]).shape} dtype={np.asarray(s0[k]).dtype}")
print(f"  geometry_id={s0.get('geometry_id')} theta={s0.get('theta')} traj_id={s0.get('traj_id')}")

# --- raw filtered_data + sentinel probe ---
f=glob.glob("data/random_task/hermite_distinct/position_set_00/channel_option_0.1/random/filtered_data/filtered_data.npy")[0]
filt=np.load(f).copy()                     # (tau, ppt, n_traj, n_sensors)
tau,ppt,n_traj,ns=filt.shape
print(f"\nraw filtered_data: (tau={tau}, ppt={ppt}, n_traj={n_traj}, n_sensors={ns})")
SENT=7; VAL=123456.0
filt[:,:,:,SENT]=VAL                        # mark sensor index 7's whole channel
X=P.reshape_input_data(filt)                # processor: -> (n_traj, ppt, tau*ns)
print(f"reshape_input_data -> X {X.shape}  (feat_dim={X.shape[-1]}=tau*ns={tau*ns})")
xf=t.as_tensor(X[3,11].astype(np.float32))  # one point's flat 2010-vector

# encoder reshape (from SpatialEncoder.forward): view(tau, ns).transpose -> (ns, tau)
tok=xf.view(tau,ns).transpose(0,1)          # (ns, tau)
recovered=tok[SENT]                          # should be ALL == VAL
ok=bool((recovered==VAL).all())
leak=int((tok[[i for i in range(ns) if i!=SENT]]==VAL).sum())
print(f"\n[CORRECT reshape view(tau,ns).T] sensor {SENT} slot all=={VAL}? {ok}   "
      f"leak into other sensors: {leak} elems")

# WRONG reshape: view(ns, tau) (the silent-failure the docstring warns about)
wrong=xf.view(ns,tau)                        # (ns, tau) but WRONG axis order
wrong_ok=bool((wrong[SENT]==VAL).all())
wrong_hits=int((wrong==VAL).sum())
print(f"[WRONG   reshape view(ns,tau)  ] sensor {SENT} slot all=={VAL}? {wrong_ok}   "
      f"total sentinel elems found={wrong_hits} (scrambled across rows)")

# confirm tau-major/sensor-minor formula: flat[k]=feature(tau=k//ns, sensor=k%ns)
k_all=np.where(X[3,11]==VAL)[0]
sensors_hit=set((k%ns) for k in k_all)
print(f"\nflat indices == sentinel map to sensor set {{k%ns}} = {sensors_hit}  (expect {{{SENT}}})")
print(f"=> tau-major/sensor-minor CONFIRMED: {sensors_hit=={SENT}}")

"""End-to-end: does the production path (DataLoader -> model_forward -> loss)
work under calibration, and does it REPRODUCE the independent scratch
measurement (q3_calibration.py: ORACLE-aug150 = 59.2 m at ctx=40)?"""
import os, sys, importlib.util
import numpy as np, torch as t
from torch.utils.data import DataLoader
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO); sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO,"scripts","train"))
spec = importlib.util.spec_from_file_location("T","scripts/train/train_np_geometry.py")
T = importlib.util.module_from_spec(spec); sys.modules["T"]=T; spec.loader.exec_module(T)
import src.models.anp as anp_mod

dev = t.device("cuda" if t.cuda.is_available() else "cpu")
DD = "data/random_task/hermite_shared/processed/geometry_split"
CK = "output/hydra/training/2026-07-23/spatial_cnp_random_hermite_shared_layout_ood_5e-05_aug_m150/seed_0/best.pt"
ck = t.load(CK, map_location=dev); cfg = ck["config"]

val = T.TrajectoryDataset(os.path.join(DD, "val_data.pkl"))
keep = sorted({int(s["geometry_id"]) for s in val.samples})
val.samples = [s for s in val.samples if int(s.get("traj_id",-1)) < 60]
print(f"val pool: {len(val.samples)} samples / {len(keep)} geometries")

model = anp_mod.DeterministicModel(cfg["num_hidden"], ck["feat_dim"], ck["out_dim"],
                                   dropout=cfg["dropout"], spatial_cfg=cfg["spatial"]).to(dev)
model.load_state_dict(ck["model"])
ym, ys = ck["y_mean"], ck["y_std"]

for ctx_n, ref in [(40, 59.2), (10, None)]:
    coll = T.make_collate(2, 40, fixed_ctx=ctx_n, seed=1, ctx_sample_mode="first",
                          exclude_ctx_from_target=True, calibration=True, dataset=val)
    dl = DataLoader(val, batch_size=64, shuffle=False, num_workers=0, collate_fn=coll)
    loss, nll, mae, per_geo, kl = T.run_epoch(model, "split", dl, dev, 0.0, None,
                                              desc="", y_mean=ym, y_std=ys)
    geo_mae = float(np.mean(list(per_geo.values())))
    tag = f"  (independent scratch measurement: {ref})" if ref else ""
    print(f"ctx={ctx_n:2d}  CALIBRATION val MAE = {mae:6.2f} (sample-wt) / "
          f"{geo_mae:6.2f} (geometry-wt){tag}")

# same-trajectory control through the SAME production path
coll0 = T.make_collate(2, 40, fixed_ctx=40, seed=1, ctx_sample_mode="first",
                       exclude_ctx_from_target=True, calibration=False)
dl0 = DataLoader(val, batch_size=64, shuffle=False, num_workers=0, collate_fn=coll0)
_,_,mae0,pg0,_ = T.run_epoch(model, "split", dl0, dev, 0.0, None, desc="", y_mean=ym, y_std=ys)
print(f"ctx=40  SAME-TRAJ    val MAE = {mae0:6.2f} (sample-wt) / "
      f"{float(np.mean(list(pg0.values()))):6.2f} (geometry-wt)   (scratch: 21.4)")

# gradients actually flow under calibration
model.train()
coll_t = T.make_collate(2, 40, seed=3, ctx_sample_mode="first", per_sample=True,
                        aug_m=150.0, calibration=True, dataset=val)
b = coll_t([val[i] for i in range(16)])
mean, var, loss, kl, nll = T.model_forward(model, "split", b, dev, 0.0, with_target_y=True)
loss.backward()
gn = sum(float(p.grad.norm()) for p in model.parameters() if p.grad is not None)
print(f"\nforward+backward under calibration+aug: loss={float(loss):.4f} finite={t.isfinite(loss).item()} "
      f"grad_norm_sum={gn:.3f} nonzero={gn>0}")

"""Correctness tests for the PFN backbone. Synthetic tensors only -- no data
loaded, so this is safe to run alongside a large training job."""
import sys, torch as t
sys.path.insert(0, ".")
from src.models.pfn import PFNModel

fails = []
def check(c, m):
    print(("  PASS  " if c else "  FAIL  ") + m)
    if not c: fails.append(m)

t.manual_seed(0)
B, n, m, F_, D = 4, 12, 20, 40, 3
cx, cy = t.randn(B, n, F_), t.randn(B, n, D)
tx, ty = t.randn(B, m, F_), t.randn(B, m, D)
sp = t.randn(B, 10, 3)

# --- flat (no spatial front end) ---
net = PFNModel(64, F_, D, n_layers=3, n_heads=4).eval()
mean, var, loss, kl, nll = net(cx, cy, tx, ty)
check(mean.shape == (B, m, D) and var.shape == (B, m, D), f"shapes {tuple(mean.shape)} / {tuple(var.shape)}")
check(kl is None, "kl is None (no latent variable)")
check((var > 0).all(), "variance strictly positive")
check(t.isfinite(loss) and loss.item() == nll.item(), "loss == nll, finite (plain Prior-Data NLL)")

# --- THE defining property: targets must not see each other ---
tx2 = tx.clone(); tx2[:, 5] = t.randn(B, F_)          # perturb ONE target's input
m2, _, _, _, _ = net(cx, cy, tx2, None)
m1, _, _, _, _ = net(cx, cy, tx, None)
others = [i for i in range(m) if i != 5]
check(t.allclose(m1[:, others], m2[:, others], atol=1e-6),
      "perturbing one target leaves ALL other targets unchanged (TabPFN mask)")
check(not t.allclose(m1[:, 5], m2[:, 5], atol=1e-6),
      "...while that target's own prediction does change")

# --- permutation invariance over the CONTEXT set (no positional encoding) ---
perm = t.randperm(n)
mp, _, _, _, _ = net(cx[:, perm], cy[:, perm], tx, None)
check(t.allclose(m1, mp, atol=1e-5), "output invariant to context ordering")

# --- context influences the prediction at all ---
mc, _, _, _, _ = net(t.randn_like(cx), cy, tx, None)
check(not t.allclose(m1, mc, atol=1e-5), "context actually conditions the output")

# --- masks ---
cmask = t.ones(B, n, dtype=t.bool); cmask[:, 6:] = False
mm1, _, _, _, _ = net(cx, cy, tx, None, context_mask=cmask)
cx3 = cx.clone(); cx3[:, 6:] = 999.0                  # garbage in the masked slots
mm2, _, _, _, _ = net(cx3, cy, tx, None, context_mask=cmask)
check(t.allclose(mm1, mm2, atol=1e-5), "masked context slots are genuinely ignored")
tmask = t.zeros(B, m, dtype=t.bool); tmask[:, :4] = True
_, _, l_masked, _, _ = net(cx, cy, tx, ty, target_mask=tmask)
check(t.isfinite(l_masked), "target_mask scoring path finite")
edge = t.zeros(B, n, dtype=t.bool); edge[:, 0] = True
_, _, _, _, nll_e = net(cx, cy, tx, ty, context_mask=edge)
check(t.isfinite(nll_e), "single-valid-context-point edge case is finite (no NaN)")

# --- spatial front end + gradients ---
sn = PFNModel(64, F_, D, n_layers=3, n_heads=4,
              spatial_cfg={"enabled": True, "n_sensors": 10, "use_position": True,
                           "norm_acoustic": "none", "n_attn_layers": 1})
sn.train()
_, _, sl, _, _ = sn(cx, cy, tx, ty, sensor_pos=sp)
sl.backward()
gn = sum(float(p.grad.norm()) for p in sn.parameters() if p.grad is not None)
check(t.isfinite(sl) and gn > 0, f"spatial front end forward+backward, grad norm {gn:.2f}")

# --- the point of the exercise: the FFN the old block lacks ---
import src.models.anp as anp
p_pfn = sum(p.numel() for p in PFNModel(128, F_, D, n_layers=6, n_heads=8).parameters())
p_cnp = sum(p.numel() for p in anp.DeterministicModel(128, F_, D).parameters())
has_ffn = any("ffn" in k for k, _ in PFNModel(64, F_, D, n_layers=2).named_parameters())
check(has_ffn, "PFN block HAS a feed-forward sublayer")
check(not any("ffn" in k for k, _ in anp.DeterministicModel(64, F_, D).named_parameters()),
      "...which the existing anp.Attention block does not")
print(f"\n  params: PFN(6L,128) {p_pfn/1e6:.2f}M vs CNP(128) {p_cnp/1e6:.2f}M")
print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)

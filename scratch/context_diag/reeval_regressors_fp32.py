"""Re-evaluate the no-context regressors in float32 (section 13.9): all 50 points of
every held-out transit, intact and with 3 / 5 / 7 sensors lost (and position error),
through acoustic_regressor.robustness. Writes results_<tag>_robust_fp32.csv beside
the checkpoint; the bfloat16 results_<tag>_robust.csv files are left as they were.

    python scratch/context_diag/reeval_regressors_fp32.py <b1 cache> <b2 cache>
"""
import os, sys
from types import SimpleNamespace
import numpy as np
import torch as t
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scratch", "context_diag")); sys.path.insert(0, os.path.join(REPO, "scripts", "eval")); sys.path.insert(0, REPO)
import acoustic_regressor as R      # noqa: E402
import regressor_env as RE          # noqa: E402

B1 = "data/pfn10kd/hermite_distinct/processed/geometry_split"
B2 = "data/pathD/hermite_distinct/processed/geometry_split"
ENV = ["h0", "hr0", "c", "c2f", "k"]
#        benchmark, tag,                    feature, loss,   environment keys
RUNS = [("b2", "acoustic_tone_nll_drop",        "tone", "nll",    None),
        ("b2", "acoustic_tone_nll_drop_s1",     "tone", "nll",    None),
        ("b2", "acoustic_raw_nll_drop",         "raw",  "nll",    None),
        ("b2", "acoustic_tone_nll_drop_env",    "tone", "nll",    ENV),
        ("b2", "acoustic_tone_nll_drop_env_s1", "tone", "nll",    ENV),
        ("b1", "acoustic_raw",                  "raw",  "euclid", None),
        ("b1", "acoustic_raw_s1",               "raw",  "euclid", None),
        ("b1", "acoustic_tone",                 "tone", "euclid", None)]


def pools(cache, feat, dev, env):
    stats = None
    if feat == "tone":
        la = t.as_tensor(np.load(os.path.join(cache, "train.npz"))["tone"]).abs().clamp_min(1e-12).log()
        stats = (float(la.mean()), float(la.std()))
    ho = [R.load_pool(cache, p, feat, dev, stats, fdev=dev if feat == "tone" else t.device("cpu"))
          for p in ("val", "test")]
    if env:
        T = RE.env_table(env, False)
        for D in ho:
            ns = D["F"].shape[1]
            e = t.as_tensor(T[D["gid"]], dtype=D["F"].dtype, device=D["F"].device)
            D["F"] = t.cat([D["F"], e[:, None, None, :].expand(D["N"], D["ppt"], ns, len(env))
                            .reshape(D["N"] * D["ppt"], ns, len(env))], -1)
    return ho


def main():
    caches = {"b1": sys.argv[1], "b2": sys.argv[2]}
    dev = t.device("cuda")
    for bench, tag, feat, loss, env in RUNS:
        out = os.path.join(REPO, "output/eval_paper/baselines", bench)
        A = SimpleNamespace(data_dir=os.path.join(REPO, B1 if bench == "b1" else B2), encoder="set",
                            pos_code="raw", frame="centroid", aug="rot", d=256, layers=6, loss=loss,
                            var_floor=0.01, feat=feat, tokenizer="mlp", no_ffn=False, summary="readout",
                            pos_err=0.0, trans_m=150.0)
        ho = pools(caches[bench], feat, dev, env)
        net = R.Net(A, ho[0]["F"].shape[-1], ho[0]["sp"].shape[1]).to(dev)
        sd = t.load(os.path.join(out, tag + ".pt"), map_location=dev)
        if not any(k.startswith("body.") for k in sd):       # saved before Net wrapped the body
            sd = {"body." + k: v for k, v in sd.items()}
        net.load_state_dict(sd); net.eval()
        print(f"[{bench} {tag}] float32", flush=True)
        R.robustness(net, ho, A, os.path.join(out, f"results_{tag}_robust_fp32.csv"))
        del net, ho


if __name__ == "__main__":
    main()

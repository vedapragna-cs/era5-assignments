"""E21 analysis -- paired comparison of two arms over the validation set.

WHY NOT JUST COMPARE SEED MEANS.  With 3 seeds per arm, "18.3x the seed spread" is not a
test and n=3 supports no useful interval.  The per-token loss vectors let us compare arms
on the SAME validation tokens, which removes val-set sampling from the comparison and
raises n from 3 to 204,544.

WHY A BLOCK BOOTSTRAP.  Validation tokens are contiguous natural text, so per-token losses
are strongly autocorrelated; an i.i.d. bootstrap would badly understate the interval.  This
uses a moving-block bootstrap with block length L (default 512 = 2 x the training sequence
length), which preserves local dependence.

BOTH numbers are reported, because they answer different questions:
  * block-bootstrap CI  -> is this difference real for THESE trained models?
  * across-seed spread  -> does it survive re-training from a different initialisation?
A claim needs both.  The first without the second is a claim about two particular runs.
"""
import numpy as np, glob, os, sys, json

def load(job_arm_glob, root="results/pertok"):
    fs = sorted(glob.glob(os.path.join(root, job_arm_glob)))
    if not fs: raise FileNotFoundError(job_arm_glob)
    return fs, [np.load(f).astype(np.float64) for f in fs]

def block_boot(d, L=512, B=2000, seed=0):
    """moving-block bootstrap of the mean of d"""
    g = np.random.default_rng(seed); n = len(d)
    nb = int(np.ceil(n / L)); starts_max = n - L
    out = np.empty(B)
    for b in range(B):
        s = g.integers(0, starts_max + 1, size=nb)
        out[b] = np.mean(np.concatenate([d[i:i+L] for i in s])[:n])
    return out

def compare(name_a, glob_a, name_b, glob_b, L=512, B=2000):
    fa, A = load(glob_a); fb, Bv = load(glob_b)
    ma = np.array([x.mean() for x in A]); mb = np.array([x.mean() for x in Bv])
    print(f"\n=== {name_a}  vs  {name_b} ===")
    print(f"  {name_a:<18} n={len(A)} seed means {np.round(ma,4)}  mean {ma.mean():.4f}")
    print(f"  {name_b:<18} n={len(Bv)} seed means {np.round(mb,4)}  mean {mb.mean():.4f}")
    diff = mb.mean() - ma.mean()
    print(f"  difference (B - A) = {diff:+.4f} nats")
    # seed-level: every cross-seed pairing, so the interval reflects init variance
    cross = np.array([y - x for x in ma for y in mb])
    print(f"  across-seed: all {len(cross)} pairings span [{cross.min():+.4f}, {cross.max():+.4f}]"
          f"   sign consistent: {bool(np.all(np.sign(cross) == np.sign(diff)))}")
    # token-level paired block bootstrap, averaged over every cross-seed pairing
    dd = np.mean(np.stack([Bv[j] - A[i] for i in range(len(A)) for j in range(len(Bv))]), axis=0)
    bs = block_boot(dd, L=L, B=B)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  paired block bootstrap (L={L}, B={B}): mean {dd.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  excludes zero: {bool(lo > 0 or hi < 0)}")
    return dict(a=name_a, b=name_b, diff=float(diff), ci=[float(lo), float(hi)],
                seed_means_a=ma.tolist(), seed_means_b=mb.tolist(),
                sign_consistent=bool(np.all(np.sign(cross) == np.sign(diff))))

def fp16_adequacy():
    """per-token losses are stored float16; confirm that costs nothing at the mean."""
    g = np.random.default_rng(0); x = g.normal(6.0, 3.0, 204544).clip(0.01, 25)
    err = x.astype(np.float16).astype(np.float64).mean() - x.mean()
    print(f"float16 storage check: mean shifts by {err:+.2e} nats over 204,544 tokens "
          f"(claims are quoted to 1e-4)")

if __name__ == "__main__":
    fp16_adequacy()
    PAIRS = [("byte","013_e21a_byte_byte_s*.npy","byte_shuf","014_e21b_byteshuf_byte_shuf_s*.npy"),
             ("bigram_pos","015_e21c_bigrampos_bigram_pos_s*.npy",
              "bigram_pos_shuf","016_e21d_bigramposshuf_bigram_pos_shuf_s*.npy"),
             ("byte","013_e21a_byte_byte_s*.npy","rand4","013_e21a_byte_rand4_s*.npy"),
             ("byte","013_e21a_byte_byte_s*.npy","dense_zn","014_e21b_byteshuf_dense_zn_s*.npy")]
    out = []
    for a, ga, b, gb in PAIRS:
        try: out.append(compare(a, ga, b, gb))
        except FileNotFoundError as e: print(f"\n=== {a} vs {b} === not available yet ({e})")
    if out: json.dump(out, open("e21_paired_results.json","w"), indent=1)

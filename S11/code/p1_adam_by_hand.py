"""S11 Part 1 + Part 2 - Adam by hand, and what bias correction actually costs.

Part 1: one weight, five gradients. Compute m, v, m_hat, v_hat and the step by hand,
        then check every intermediate against torch.optim.Adam.
Part 2: disable bias correction; report when the difference stops mattering.

No GPU. Deterministic. Run: python3 p1_adam_by_hand.py
"""
import json, math, torch

B1, B2, EPS, LR = 0.9, 0.999, 1e-8, 1e-3
W0    = 0.7000
GRADS = [0.5000, -0.3000, 0.8000, 0.1000, -0.6000]

# ---------------------------------------------------------------- Part 1: by hand
print("=" * 78)
print("PART 1 - Adam by hand vs torch.optim.Adam")
print(f"  w0={W0}  lr={LR}  betas=({B1},{B2})  eps={EPS}  weight_decay=0")
print("=" * 78)

m = v = 0.0
w = W0
hand = []
for t, g in enumerate(GRADS, start=1):
    m = B1 * m + (1 - B1) * g            # first moment  (EMA of g)
    v = B2 * v + (1 - B2) * g * g        # second moment (EMA of g^2)
    bc1 = 1 - B1 ** t                    # bias corrections: E[m_t] = (1-b1^t) E[g]
    bc2 = 1 - B2 ** t
    mh, vh = m / bc1, v / bc2
    step = LR * mh / (math.sqrt(vh) + EPS)
    w = w - step
    hand.append(dict(t=t, g=g, m=m, v=v, bc1=bc1, bc2=bc2, mhat=mh, vhat=vh, step=step, w=w))

# torch, same numbers, float64 so the comparison tests the algebra not the dtype
p = torch.tensor([W0], dtype=torch.float64, requires_grad=True)
opt = torch.optim.Adam([p], lr=LR, betas=(B1, B2), eps=EPS, weight_decay=0.0)
tor = []
for t, g in enumerate(GRADS, start=1):
    opt.zero_grad(); p.grad = torch.tensor([g], dtype=torch.float64); opt.step()
    st = opt.state[p]
    tor.append(dict(t=t, m=st["exp_avg"].item(), v=st["exp_avg_sq"].item(), w=p.item()))

print(f"\n{'t':>2}{'g':>8}{'m':>13}{'v':>15}{'m_hat':>13}{'v_hat':>15}{'step':>13}{'w':>12}")
for h in hand:
    print(f"{h['t']:>2}{h['g']:>8.4f}{h['m']:>13.8f}{h['v']:>15.10f}"
          f"{h['mhat']:>13.8f}{h['vhat']:>15.10f}{h['step']:>13.4e}{h['w']:>12.8f}")

print(f"\n{'t':>2}{'|dm|':>12}{'|dv|':>12}{'|dw|':>12}   (hand vs torch)")
worst = 0.0
for h, o in zip(hand, tor):
    dm, dv, dw = abs(h['m']-o['m']), abs(h['v']-o['v']), abs(h['w']-o['w'])
    worst = max(worst, dm, dv, dw)
    print(f"{h['t']:>2}{dm:>12.2e}{dv:>12.2e}{dw:>12.2e}")
print(f"\n  worst disagreement anywhere: {worst:.3e}   -> {'MATCH' if worst < 1e-12 else 'MISMATCH'}")
print(f"  final w: hand {hand[-1]['w']:.12f} | torch {tor[-1]['w']:.12f}")

# ------------------------------------------------- Part 2: bias correction on/off
print("\n" + "=" * 78)
print("PART 2 - bias correction on vs off")
print("=" * 78)

def ratio(t):
    """step_uncorrected / step_corrected, exactly, for any gradient sequence whose
    m/sqrt(v) is the same in both runs -- i.e. the correction factors alone."""
    return (1 - B1 ** t) / math.sqrt(1 - B2 ** t)

# empirical: same weight, same gradient stream, 20 steps, correction on and off
G20 = [0.5, -0.3, 0.8, 0.1, -0.6, 0.4, -0.2, 0.7, -0.5, 0.3,
       0.6, -0.4, 0.2, -0.7, 0.5, 0.1, -0.3, 0.8, -0.6, 0.4]
def run(correct, n):
    m = v = 0.0; w = W0; out = []
    for t, g in enumerate((G20 * (n // 20 + 1))[:n], start=1):
        m = B1 * m + (1 - B1) * g
        v = B2 * v + (1 - B2) * g * g
        if correct: mh, vh = m / (1 - B1 ** t), v / (1 - B2 ** t)
        else:       mh, vh = m, v
        s = LR * mh / (math.sqrt(vh) + EPS)
        w -= s; out.append((s, w))
    return out

on20, off20 = run(True, 20), run(False, 20)
print(f"\n{'t':>3}{'step ON':>13}{'step OFF':>13}{'OFF/ON':>9}{'w ON':>12}{'w OFF':>12}")
for t in range(20):
    so, sf = on20[t][0], off20[t][0]
    print(f"{t+1:>3}{so:>13.4e}{sf:>13.4e}{sf/so:>9.3f}{on20[t][1]:>12.7f}{off20[t][1]:>12.7f}")

print("\n  The ratio is not a decaying transient. It RISES first:")
print(f"  {'t':>7}{'analytic OFF/ON':>18}")
for t in (1, 2, 5, 10, 20, 50, 100, 300, 1000, 3000, 6000, 10000):
    print(f"  {t:>7}{ratio(t):>18.4f}")
pk = max(range(1, 20001), key=ratio)
print(f"\n  peak ratio {ratio(pk):.4f} at t={pk}")
for thr in (1.10, 1.05, 1.01):
    tt = next(t for t in range(1, 200001) if ratio(t) < thr)
    print(f"  first step where OFF/ON < {thr:.2f}: t={tt}")

json.dump(dict(hand=hand, torch=tor, worst=worst,
               on20=[s for s, _ in on20], off20=[s for s, _ in off20],
               ratio={t: ratio(t) for t in (1,2,5,10,20,50,100,300,1000,3000,6000,10000)},
               peak_t=pk, peak_ratio=ratio(pk),
               thresholds={f"{thr}": next(t for t in range(1,200001) if ratio(t) < thr)
                           for thr in (1.10, 1.05, 1.01)}),
          open("../results/p1_adam.json", "w"), indent=1)
print("\nwrote ../results/p1_adam.json")

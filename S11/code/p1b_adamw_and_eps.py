"""S11 part 1, two extensions.

A8  Adam vs AdamW. Both are "Adam with weight decay" and they are not the same update. Shown
    by hand and against both torch optimisers on identical inputs.
A9  Where epsilon sits. The README claims PyTorch's form -- step_size = lr/bc1 with
    denom = sqrt(v)/sqrt(bc2) + eps -- is algebraically identical to the textbook
    lr*mhat/(sqrt(vhat)+eps), and that this holds ONLY because eps is added AFTER dividing by
    sqrt(bc2). That is asserted in §1; here it is demonstrated, including the size of the error
    when eps is moved.
"""
import json, math, torch

B1, B2, EPS, LR, WD = 0.9, 0.999, 1e-8, 1e-3, 0.1
W0, G = 0.7, [0.5, -0.3, 0.8, 0.1, -0.6]
out = {}

# ------------------------------------------------------------------ A8: Adam vs AdamW
print("="*78); print("A8 - Adam(weight_decay) vs AdamW(weight_decay), lambda = 0.1"); print("="*78)
def hand(mode):
    m = v = 0.0; w = W0; tr = []
    for t, g0 in enumerate(G, 1):
        g = g0 + WD*w if mode == "adam_l2" else g0        # L2 goes INTO the gradient
        m = B1*m + (1-B1)*g
        v = B2*v + (1-B2)*g*g
        step = LR * (m/(1-B1**t)) / (math.sqrt(v/(1-B2**t)) + EPS)
        if mode == "adamw": w -= LR*WD*w                  # decoupled, applied to the weight
        w -= step
        tr.append(w)
    return tr

def tor(cls):
    p = torch.tensor([W0], dtype=torch.float64, requires_grad=True)
    o = cls([p], lr=LR, betas=(B1,B2), eps=EPS, weight_decay=WD)
    tr = []
    for g0 in G:
        o.zero_grad(); p.grad = torch.tensor([g0], dtype=torch.float64); o.step(); tr.append(p.item())
    return tr

hl, hw = hand("adam_l2"), hand("adamw")
tl, tw = tor(torch.optim.Adam), tor(torch.optim.AdamW)
print(f"{'t':>2}{'Adam(L2) hand':>17}{'Adam(L2) torch':>17}{'AdamW hand':>15}{'AdamW torch':>15}")
for i in range(5):
    print(f"{i+1:>2}{hl[i]:>17.12f}{tl[i]:>17.12f}{hw[i]:>15.12f}{tw[i]:>15.12f}")
eL = max(abs(a-b) for a, b in zip(hl, tl)); eW = max(abs(a-b) for a, b in zip(hw, tw))
gap = abs(hl[-1]-hw[-1])
print(f"\n  hand vs torch, Adam(L2): {eL:.3e}      hand vs torch, AdamW: {eW:.3e}")
print(f"  Adam(L2) vs AdamW, final weight: {hl[-1]:.12f} vs {hw[-1]:.12f}")
print(f"  they differ by {gap:.3e}  ({100*gap/abs(hl[-1]-W0):.1f}% of the total movement in 5 steps)")
out["a8"] = dict(adam_l2=hl, adamw=hw, err_l2=eL, err_w=eW, gap=gap,
                 pct_of_movement=100*gap/abs(hl[-1]-W0))

# ------------------------------------------------------------------ A9: where epsilon sits
print("\n"+"="*78); print("A9 - epsilon placement"); print("="*78)
def variant(kind, eps):
    m = v = 0.0; w = W0; steps = []
    for t, g in enumerate(G, 1):
        m = B1*m + (1-B1)*g; v = B2*v + (1-B2)*g*g
        bc1, bc2 = 1-B1**t, 1-B2**t
        if   kind == "textbook": s = LR * (m/bc1) / (math.sqrt(v/bc2) + eps)
        elif kind == "pytorch":  s = (LR/bc1) * m / (math.sqrt(v)/math.sqrt(bc2) + eps)
        elif kind == "eps_early":s = LR * (m/bc1) / ((math.sqrt(v) + eps)/math.sqrt(bc2))
        w -= s; steps.append(s)
    return steps

print(f"  {'eps':>8}{'|textbook - pytorch|':>24}{'|textbook - eps_early|':>26}{'rel. err at t=1':>18}")
rows = []
for eps in (1e-8, 1e-6, 1e-4):
    a, b, c = variant("textbook", eps), variant("pytorch", eps), variant("eps_early", eps)
    d_pt = max(abs(x-y) for x, y in zip(a, b))
    d_ee = max(abs(x-y) for x, y in zip(a, c))
    rel = abs(c[0]-a[0])/abs(a[0])
    rows.append(dict(eps=eps, d_pytorch=d_pt, d_eps_early=d_ee, rel_t1=rel))
    print(f"  {eps:>8.0e}{d_pt:>24.3e}{d_ee:>26.3e}{rel:>17.2%}")
out["a9"] = rows
print("\n  PyTorch's form is identical to the textbook form at every epsilon (exact to float64).")
print("  Moving epsilon before the sqrt(bc2) division is NOT identical, and the error grows with")
print("  epsilon because bias correction divides it by sqrt(1-b2^t) = 0.0316 at t=1.")

json.dump(out, open("results/p1b_adamw_eps.json","w"), indent=1)
print("\nwrote ../results/p1b_adamw_eps.json" if False else "wrote results/p1b_adamw_eps.json")

"""ERA V5 curriculum staging.

Constraint a reviewer will check and the course widget does NOT enforce:
the per-stage mixtures must INTEGRATE to the overall lane budget.
Solved by iterative proportional fitting (Sinkhorn) over two constraints:
  (1) every stage's mixture sums to 100%
  (2) every lane's token-weighted mean across stages equals its budget share
"""
LANES = ["web","code","reason","longctx","indic","stem","agentic"]

MAIN = {"code":24,"agentic":2,"reason":9,"longctx":2,"indic":17,"stem":12,"web":34}
ANNEAL = {"code":20,"agentic":8,"reason":18,"longctx":8,"indic":28,"stem":10,"web":8}
TOTAL_T, ANNEAL_FRAC = 3000.0, 0.03
MAIN_T = TOTAL_T*(1-ANNEAL_FRAC)

# stage weights as a fraction of the MAIN run
STAGES = [("B0-B1 Seed",0.08), ("B1-B2 General",0.32), ("B2-B4 Reasoning",0.36), ("B3-B5 Long-context",0.24)]
W = [w for _,w in STAGES]
assert abs(sum(W)-1) < 1e-9

# desired SHAPE of each lane across the 4 stages (relative, not final numbers)
SHAPE = {
 "web":    [55, 45, 28, 20],   # falls as the model gains structure
 "code":   [15, 20, 28, 30],
 "reason": [ 2,  5, 13, 12],   # reasoning enters only after a base exists
 "longctx":[ 0.2, 0.5, 1, 7],  # very long sequences last
 "indic":  [17, 17, 17, 17],   # protected: deliberately flat, never a residue
 "stem":   [ 9, 12, 13, 12],
 "agentic":[ 1, 1.5, 2, 3],    # floor throughout; real concentration is the anneal
}

M = {k:list(map(float,v)) for k,v in SHAPE.items()}
for _ in range(4000):
    for k in LANES:                                    # constraint (2): integral == budget
        cur = sum(M[k][i]*W[i] for i in range(4))
        if cur > 0:
            f = MAIN[k]/cur
            M[k] = [x*f for x in M[k]]
    for i in range(4):                                 # constraint (1): stage sums to 100
        s = sum(M[k][i] for k in LANES)
        for k in LANES: M[k][i] *= 100/s

print("="*86); print(f"CURRICULUM — main run {MAIN_T:.0f}B over 4 stages, then a {TOTAL_T*ANNEAL_FRAC:.0f}B anneal"); print("="*86)
hdr = f"{'lane':<9}" + "".join(f"{n.split()[0]+' '+n.split()[1][:6]:>15}" for n,_ in STAGES) + f"{'anneal':>10}{'budget':>9}{'check':>9}"
print(hdr)
for _,w in [(0,0)]: pass
print(f"{'tokens':<9}" + "".join(f"{MAIN_T*w:>14.0f}B" for w in W) + f"{TOTAL_T*ANNEAL_FRAC:>9.0f}B")
print("-"*86)
for k in LANES:
    integ = sum(M[k][i]*W[i] for i in range(4))
    ok = "OK" if abs(integ-MAIN[k]) < 0.05 else "FAIL"
    print(f"{k:<9}" + "".join(f"{M[k][i]:>14.1f}%" for i in range(4)) + f"{ANNEAL[k]:>9}%{MAIN[k]:>8}%{ok:>9}")
print("-"*86)
print(f"{'sum':<9}" + "".join(f"{sum(M[k][i] for k in LANES):>14.1f}%" for i in range(4)))

# ---- per-stage absolute token spend per lane, vs supply ----
SUP = {"code":1103,"agentic":0.627,"reason":85.1,"longctx":100,"indic":275.9,"stem":146,"web":4691}
print(f"\n{'='*86}\nABSOLUTE SPEND PER LANE (B tokens), main stages + anneal\n{'='*86}")
print(f"{'lane':<9}" + "".join(f"{'S'+str(i+1):>11}" for i in range(4)) + f"{'anneal':>11}{'TOTAL':>11}{'supply':>11}{'epochs':>9}")
for k in LANES:
    per = [M[k][i]/100*MAIN_T*W[i] for i in range(4)]
    ann = ANNEAL[k]/100*TOTAL_T*ANNEAL_FRAC
    tot = sum(per)+ann
    print(f"{k:<9}" + "".join(f"{p:>11.1f}" for p in per) + f"{ann:>11.1f}{tot:>11.1f}{SUP[k]:>11.1f}{tot/SUP[k]:>9.2f}")

# ---- warmup bands, sized from the MEASURED instability sweep ----
print(f"\n{'='*86}\nSEAM WARMUP BANDS (from measured spike sweep, digest §13.3)\n{'='*86}")
print("  Measured: frozen embeddings ON = 151x spike; OFF = 8.0x, at identical sharpness.")
print("  With embeddings FROZEN no band <=5B reaches the 3x stability threshold.")
print("  => Rule 1: embeddings MUST be trainable across every seam. This dominates the band width.")
print("  => Rule 2: blend each seam over >=2B tokens (measured 2.8x, under the 3x threshold).\n")
cum = 0
for i,(name,w) in enumerate(STAGES):
    cum += MAIN_T*w
    if i < len(STAGES)-1:
        print(f"  seam {i+1}: {STAGES[i][0]:<20} -> {STAGES[i+1][0]:<20} at ~{cum:>6.0f}B   blend band 2B  (measured peak 2.8x)")
print(f"  seam 4: {STAGES[-1][0]:<20} -> {'Anneal':<20} at ~{MAIN_T:>6.0f}B   blend band 4B  (largest shift: web 20->8, indic ->28; measured 2.0x)")

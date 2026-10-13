"""
E17 — rank predictor for codec ("headless") output heads at V = 50,257 / 131,072 / 1,000,000.

A codec head replaces the dense V x d output matrix with
        logits_v = < phi(v) , W h >,        W : R^d -> R^D,  phi(v) in {0,1}^D fixed
so head params = d*D, independent of V.  That is the whole point of assignment P5
("get rid of the final head... vocab of 1M without any issues").

Writing S in {0,1}^{V x D} for the stacked phi(v), the achievable logit vectors are exactly
col-space(S).  Three exact consequences, none of which need a trained model:

  1. rank(S)  -- the head can produce at most rank(S) independent logit patterns over V tokens.
                 Hard cap: rank(S) <= D, INDEPENDENT OF V.
  2. collisions -- tokens with identical rows of S can never be separated by any h.
                 Directly bounds the "100% decodable" claim.
  3. unigram floor -- min over u of CE(softmax(S u), p) - H(p), in nats/token.
                 Convex, solved to optimality.  A LOWER bound on the head's true cost:
                 a real model must also produce good *conditional* distributions.
"""
import numpy as np, scipy.sparse as sp, json, sys, unicodedata
from scipy.optimize import minimize
from scipy.special import logsumexp

# ------------------------------------------------------------------ codecs --
def cells_byte(tokstr, P):
    b = tokstr.encode("utf-8", "replace")[:P]
    return [int(v)*P + i for i, v in enumerate(b)], 256*P

def cells_char(tokstr, P):
    NB = 1114112 // 128 + 2                      # unicode blocks of 128 + 2 reserved
    out = []
    for i, ch in enumerate(tokstr[:P]):
        cp = ord(ch); blk, off = cp >> 7, cp & 0x7F
        out.append(blk*P + i)                    # block family
        out.append(NB*P + off*P + i)             # offset family
    return out, NB*P + 128*P

BRAHMIC_LO, BRAHMIC_HI = 0x0900, 0x0D80
def cells_phoneme(tokstr, P):
    """E16.2: for Brahmic codepoints, cp & 0x7F is a script-invariant phoneme id
    (76/124 offsets carry an identical phoneme label across all nine scripts).
    Give every Brahmic script ONE shared phoneme family, plus a 16-way script tag
    that is NOT position-bound (so it costs 16 cells, not 16*P)."""
    out = []
    NSCRIPT, PHON, NOTH = 16, 128, 256
    base_ph  = 0
    base_sc  = PHON*P
    base_oth = base_sc + NSCRIPT
    for i, ch in enumerate(tokstr[:P]):
        cp = ord(ch)
        if BRAHMIC_LO <= cp < BRAHMIC_HI:
            out.append(base_ph + (cp & 0x7F)*P + i)       # shared across all 9 scripts
            out.append(base_sc + ((cp - BRAHMIC_LO) >> 7))  # script tag, NOT position-bound
        else:
            for bv in chr(cp).encode("utf-8","replace"):    # non-Brahmic: byte fallback
                out.append(base_oth + int(bv)*P + i)
    return out, base_oth + NOTH*P

def cells_factored(idx, V, nfac):
    """Index-factored head (classic factored softmax / the other plausible 'K2'):
    v -> (v1,...,vk) in a mixed radix; logits_v = sum of k learned scalars.
    D = k * ceil(V**(1/k)).  Surface form plays no role."""
    r = int(np.ceil(V ** (1.0/nfac)))
    out, x = [], idx
    for j in range(nfac):
        out.append(j*r + (x % r)); x //= r
    return out, nfac*r

# ------------------------------------------------------------- build S ------
def build_S(tokens, codec, P=32, nfac=None):
    rows, cols, D = [], [], None
    for i, t in enumerate(tokens):
        if codec == "factored": c, D = cells_factored(i, len(tokens), nfac)
        elif codec == "byte":   c, D = cells_byte(t, P)
        elif codec == "char":   c, D = cells_char(t, P)
        elif codec == "phon":   c, D = cells_phoneme(t, P)
        elif codec == "bigram": c, D = cells_bigram(t, P)
        # CONVENTION (see CORRECTION C1): the trained model computes W[idx].sum(1), so a cell
        # occurring twice in a token contributes twice.  set() here is the DEDUP convention and
        # is retained only so the committed E17 tables remain reproducible as published; it makes
        # ranks for arms with a position-free family (bigram32, phon32) high by <1%.
        c = sorted(set(c))
        rows.extend([i]*len(c)); cols.extend(c)
    S = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                      shape=(len(tokens), D))
    S.data[:] = 1.0                                  # dedup already done per row
    return S

def compact(S):
    """drop all-zero columns; rank and the optimisation only ever see reachable cells"""
    used = np.diff(S.tocsc().indptr) > 0
    return S[:, used], int(used.sum())

# ------------------------------------------------------------- measures -----
def rank_of(S, tol=1e-9):
    G = (S.T @ S).toarray().astype(np.float64)
    ev = np.linalg.eigvalsh(G)
    return int((ev > ev.max()*tol).sum()), ev

def collisions(S):
    """tokens whose row of S is identical to another token's -> provably indistinguishable"""
    Sc = S.tocsr()
    keys = [hash(Sc.indices[Sc.indptr[i]:Sc.indptr[i+1]].tobytes()) for i in range(Sc.shape[0])]
    _, counts = np.unique(np.array(keys, dtype=np.int64), return_counts=True)
    ndistinct = len(counts)
    return Sc.shape[0] - ndistinct, ndistinct, int(counts.max())

def unigram_floor(S, freq, maxiter=400):
    """min_u  CE(softmax(S u), p) - H(p).  Convex in u.  Returns extra nats/token."""
    p = freq.astype(np.float64); p = p / p.sum()
    H = -(p[p > 0] * np.log(p[p > 0])).sum()
    Sd = S.astype(np.float64).tocsr(); St = Sd.T.tocsr()
    def fg(u):
        z = Sd @ u
        lse = logsumexp(z)
        f = lse - p @ z
        q = np.exp(z - lse)
        return f, St @ (q - p)
    r = minimize(fg, np.zeros(S.shape[1]), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=maxiter, maxfun=maxiter*2, ftol=1e-14, gtol=1e-10))
    return float(r.fun - H), float(H), bool(r.success), int(r.nit)

def cells_bigram(tokstr, P, NBIG=8192):
    """Axis C: kappa is LINEAR in phi, so 'ab' at (1,2) is exactly emb(a@1)+emb(b@2)
    and no conjunction is representable at any training budget.  Add position-free
    hashed adjacent-byte bigrams: the minimal conjunction, and shift-invariant.
    D = 256*P + NBIG, matched to byte64 at P=32."""
    b = tokstr.encode("utf-8","replace")[:P]
    out = [int(v)*P + i for i, v in enumerate(b)]
    base = 256*P
    for i in range(len(b)-1):
        out.append(base + ((int(b[i])*257 + int(b[i+1])*1013904223) & 0x7FFFFFFF) % NBIG)
    return out, base + NBIG

"""
zero_sim.py - 32 virtual GPUs + DDP / ZeRO-1 / ZeRO-2 / ZeRO-3, in plain PyTorch on CPU.

Every "GPU" is a VirtualGPU object that owns real tensors. Memory is measured by summing
the bytes of the tensors each GPU holds at each moment (not by a formula), so the numbers
we compare against the ZeRO paper come from what the code actually stores.

Mixed precision follows the ZeRO paper:
    params (bf16) 2 bytes + grads (bf16) 2 bytes + Adam states in fp32 (master 4 + m 4 + v 4) = 16 bytes / param
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------
# 1. Virtual GPUs
# ----------------------------------------------------------------------------------------
PARAM_DTYPE = torch.bfloat16    # 2 bytes
GRAD_DTYPE = torch.bfloat16     # 2 bytes
MASTER_DTYPE = torch.float32    # 4 bytes each for master weights, Adam m, Adam v

CATEGORIES = ["params", "grads", "optim", "activations", "gathered_params", "grad_temp"]


class OutOfMemory(RuntimeError):
    pass


class VirtualGPU:
    """A fake GPU: a named tensor store with byte-exact memory accounting."""

    def __init__(self, rank: int, node: int, capacity_bytes: int | None = None):
        self.rank, self.node, self.capacity = rank, node, capacity_bytes
        self.store: dict[str, tuple[str, torch.Tensor]] = {}
        self.used = {c: 0 for c in CATEGORIES}
        self.peak_total = 0
        self.peak_by_cat = dict(self.used)
        self.track = False            # record a memory timeline (used for rank 0 plots)
        self.timeline: list[dict] = []

    # --- allocation --------------------------------------------------------------------
    def alloc(self, name: str, tensor: torch.Tensor, category: str) -> torch.Tensor:
        if name in self.store:
            self.free(name)
        self.store[name] = (category, tensor)
        self.used[category] += tensor.numel() * tensor.element_size()
        self._update(f"+{name}")
        return tensor

    def free(self, name: str):
        cat, t = self.store.pop(name)
        self.used[cat] -= t.numel() * t.element_size()
        self._update(f"-{name}")

    def get(self, name: str) -> torch.Tensor:
        return self.store[name][1]

    @property
    def total(self) -> int:
        return sum(self.used.values())

    def model_state_bytes(self) -> int:
        return self.used["params"] + self.used["grads"] + self.used["optim"]

    def _update(self, label: str):
        tot = self.total
        if self.capacity is not None and tot > self.capacity:
            raise OutOfMemory(
                f"GPU{self.rank}: tried to hold {tot/2**20:.2f} MiB, capacity {self.capacity/2**20:.2f} MiB"
            )
        if tot > self.peak_total:
            self.peak_total, self.peak_by_cat = tot, dict(self.used)
        if self.track:
            self.timeline.append({"event": label, **self.used, "total": tot})

    def reset_peak(self):
        self.peak_total, self.peak_by_cat = self.total, dict(self.used)


class VirtualCluster:
    """N virtual GPUs split into nodes, plus NCCL-style collectives with a communication log.

    Communication volume uses ring algorithms (what NCCL uses for large messages):
        all-reduce     : each rank sends 2(N-1)/N * M bytes
        reduce-scatter : each rank sends  (N-1)/N * M bytes
        all-gather     : each rank sends  (N-1)/N * M bytes   (M = size of the full tensor)
    """

    def __init__(self, world_size=32, gpus_per_node=8, capacity_bytes=None,
                 nvlink_gbps=450.0, infiniband_gbps=50.0):
        self.N = world_size
        self.gpus_per_node = gpus_per_node
        self.gpus = [VirtualGPU(r, r // gpus_per_node, capacity_bytes) for r in range(world_size)]
        self.num_nodes = math.ceil(world_size / gpus_per_node)
        # bandwidths in GB/s (numbers used in the lecture: NVLink ~450 GB/s, InfiniBand ~10x slower)
        self.bottleneck_gbps = nvlink_gbps if self.num_nodes == 1 else infiniband_gbps
        self.comm_log: list[dict] = []
        self.phase = ""

    def __len__(self):
        return self.N

    # --- helpers -----------------------------------------------------------------------
    @staticmethod
    def _sum(tensors):
        # Same reduction order for every collective -> results are bit-identical across stages.
        acc = tensors[0].to(torch.float32).clone()
        for t in tensors[1:]:
            acc += t.to(torch.float32)
        return acc

    def _log(self, op, tag, full_numel, dtype, factor):
        elsize = torch.empty((), dtype=dtype).element_size()
        per_rank_elems = factor * (self.N - 1) / self.N * full_numel
        per_rank_bytes = per_rank_elems * elsize
        self.comm_log.append({
            "op": op, "tag": tag, "phase": self.phase, "full_numel": full_numel,
            "elems_sent_per_rank": per_rank_elems, "bytes_sent_per_rank": per_rank_bytes,
            "est_seconds": per_rank_bytes / (self.bottleneck_gbps * 1e9),
        })

    # --- collectives -------------------------------------------------------------------
    def all_reduce(self, tensors, tag="", average=True):
        """Every rank ends with the (averaged) sum of everyone's tensor."""
        assert len(tensors) == self.N
        s = self._sum(tensors)
        if average:
            s /= self.N
        self._log("all_reduce", tag, tensors[0].numel(), tensors[0].dtype, 2)
        return [s.to(t.dtype) for t in tensors]

    def reduce_scatter(self, tensors, tag="", average=True):
        """Same reduction as all-reduce, but rank r keeps only slice r of the result."""
        assert len(tensors) == self.N and tensors[0].numel() % self.N == 0
        s = self._sum(tensors)
        if average:
            s /= self.N
        self._log("reduce_scatter", tag, tensors[0].numel(), tensors[0].dtype, 1)
        return [c.to(tensors[0].dtype).clone() for c in s.chunk(self.N)]

    def all_gather(self, shards, tag=""):
        """Rank r contributes shard r; every rank ends with the concatenation of all shards."""
        assert len(shards) == self.N
        full = torch.cat(shards)
        self._log("all_gather", tag, full.numel(), full.dtype, 1)
        return [full.clone() for _ in range(self.N)]

    # --- reporting ---------------------------------------------------------------------
    def comm_summary(self, psi, steps=1):
        """Communication per rank per step in units of Psi (with the (N-1)/N factor removed)."""
        tot = sum(e["elems_sent_per_rank"] for e in self.comm_log) / steps
        return tot / psi * self.N / (self.N - 1)


# ----------------------------------------------------------------------------------------
# 2. Demo model: a residual MLP split into "units" (one unit = one layer = one FSDP/ZeRO-3 unit)
# ----------------------------------------------------------------------------------------
def make_model_spec(d_in=64, d_hidden=256, n_hidden=8, n_classes=10):
    spec = [{"name": "in_proj", "kind": "in", "shapes": [(d_hidden, d_in), (d_hidden,)]}]
    for i in range(n_hidden):
        spec.append({"name": f"block{i}", "kind": "res", "shapes": [(d_hidden, d_hidden), (d_hidden,)]})
    spec.append({"name": "head", "kind": "head", "shapes": [(n_classes, d_hidden), (n_classes,)]})
    return spec


def unit_numel(unit):
    return sum(math.prod(s) for s in unit["shapes"])


def unflatten(flat, shapes):
    out, off = [], 0
    for s in shapes:
        n = math.prod(s)
        out.append(flat[off:off + n].view(s))
        off += n
    return out


def unit_forward(kind, W, b, x):
    if kind == "in":
        return F.gelu(x @ W.T + b)
    if kind == "res":
        return x + F.gelu(x @ W.T + b)
    return x @ W.T + b  # head -> logits


def init_units(spec, world_size, seed=0):
    """Full fp32 params per unit, flattened and padded to a multiple of world_size."""
    g = torch.Generator().manual_seed(seed)
    flats = []
    for u in spec:
        parts = []
        scale = {"in": 1.0, "res": 0.5, "head": 0.1}[u["kind"]]
        for s in u["shapes"]:
            if len(s) == 2:
                parts.append(torch.randn(s, generator=g) * (scale / math.sqrt(s[1])))
            else:
                parts.append(torch.zeros(s))
        flat = torch.cat([p.flatten() for p in parts])
        pad = (-flat.numel()) % world_size
        flats.append(torch.cat([flat, torch.zeros(pad)]))
    return flats


class SyntheticData:
    """A fixed random 'teacher' network labels random inputs. Each rank gets a different micro-batch."""

    def __init__(self, d_in=64, n_classes=10, seed=123):
        g = torch.Generator().manual_seed(seed)
        self.d_in = d_in
        self.teacher = torch.randn(n_classes, d_in, generator=g)

    def batch(self, step, rank, micro_bs):
        g = torch.Generator().manual_seed(10_000 * step + rank)
        x = torch.randn(micro_bs, self.d_in, generator=g)
        y = (x @ self.teacher.T).argmax(dim=1)
        return x, y


# ----------------------------------------------------------------------------------------
# 3. Optimizer (AdamW, element-wise -> a sharded update equals the full update exactly)
# ----------------------------------------------------------------------------------------
def adamw_(master, m, v, g, step, lr, b1=0.9, b2=0.95, eps=1e-8, wd=0.0):
    m.mul_(b1).add_(g, alpha=1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    denom = (v / (1 - b2 ** step)).sqrt_().add_(eps)
    if wd:
        master.mul_(1 - lr * wd)
    master.addcdiv_(m, denom, value=-lr / (1 - b1 ** step))


# ----------------------------------------------------------------------------------------
# 4. The trainer: one class, four behaviours selected by `stage`
#    stage 0 = plain data parallel (DDP), 1 = ZeRO-1 (P_os), 2 = ZeRO-2 (P_os+g), 3 = ZeRO-3 (P_os+g+p)
# ----------------------------------------------------------------------------------------
STAGE_NAMES = {0: "DDP (no ZeRO)", 1: "ZeRO-1", 2: "ZeRO-2", 3: "ZeRO-3"}


class ZeroTrainer:
    def __init__(self, cluster: VirtualCluster, stage: int, spec=None, data=None,
                 lr=3e-3, micro_bs=16, seed=0):
        assert stage in (0, 1, 2, 3)
        self.C, self.stage = cluster, stage
        self.spec = spec or make_model_spec()
        self.data = data or SyntheticData()
        self.lr, self.micro_bs = lr, micro_bs
        self.step_num = 0
        N = cluster.N

        full = init_units(self.spec, N, seed)
        self.psi = sum(unit_numel(u) for u in self.spec)       # real parameter count
        self.psi_padded = sum(f.numel() for f in full)          # what is actually stored
        self.shard_sizes = [f.numel() // N for f in full]

        for gpu in cluster.gpus:
            r = gpu.rank
            for u, f in enumerate(full):
                shard = f.chunk(N)[r]
                # bf16 working parameters: full copy (stage 0-2) or only my shard (stage 3)
                p = f if stage < 3 else shard
                gpu.alloc(f"p{u}", p.to(PARAM_DTYPE).clone(), "params")
                # fp32 optimizer states: full (stage 0) or my shard (stage 1-3)
                o = f if stage == 0 else shard
                gpu.alloc(f"master{u}", o.to(MASTER_DTYPE).clone(), "optim")
                gpu.alloc(f"m{u}", torch.zeros_like(o, dtype=MASTER_DTYPE), "optim")
                gpu.alloc(f"v{u}", torch.zeros_like(o, dtype=MASTER_DTYPE), "optim")
                # bf16 gradient buffer: full (stage 0-1) or my shard (stage 2-3)
                gsize = f.numel() if stage < 2 else shard.numel()
                gpu.alloc(f"g{u}", torch.zeros(gsize, dtype=GRAD_DTYPE), "grads")
            gpu.reset_peak()

    # --- parameter access ----------------------------------------------------------------
    def _materialize(self, u, phase):
        """Return every rank's full bf16 params for unit u. ZeRO-3 must all-gather them first."""
        gpus = self.C.gpus
        if self.stage < 3:
            return [g.get(f"p{u}") for g in gpus]
        fulls = self.C.all_gather([g.get(f"p{u}") for g in gpus], tag=f"{phase}:params:{self.spec[u]['name']}")
        for g, t in zip(gpus, fulls):
            g.alloc(f"gathered{u}", t, "gathered_params")
        return fulls

    def _release(self, u):
        if self.stage == 3:
            for g in self.C.gpus:
                g.free(f"gathered{u}")

    # --- one training step -----------------------------------------------------------------
    def step(self):
        C, N, gpus = self.C, self.C.N, self.C.gpus
        self.step_num += 1
        units = self.spec
        batches = [self.data.batch(self.step_num, r, self.micro_bs) for r in range(N)]
        x = [b[0].to(PARAM_DTYPE) for b in batches]

        # ---------------- forward (all ranks in lock-step, unit by unit) ----------------
        C.phase = "forward"
        for u, unit in enumerate(units):
            params = self._materialize(u, "fwd")
            for r, g in enumerate(gpus):
                g.alloc(f"act{u}", x[r], "activations")       # saved input for backward
                W, b = unflatten(params[r], unit["shapes"])
                with torch.no_grad():
                    x[r] = unit_forward(unit["kind"], W, b, x[r])
            self._release(u)

        losses, grad_out = [], []
        for r in range(N):
            logits = x[r].detach().float().requires_grad_(True)
            loss = F.cross_entropy(logits, batches[r][1])
            (gl,) = torch.autograd.grad(loss, logits)
            losses.append(loss.item())
            grad_out.append(gl.to(PARAM_DTYPE))

        # ---------------- backward ----------------
        C.phase = "backward"
        for u in reversed(range(len(units))):
            unit = units[u]
            params = self._materialize(u, "bwd")
            for r, g in enumerate(gpus):
                xin = g.get(f"act{u}").detach().requires_grad_(u > 0)
                p = params[r].detach().clone().requires_grad_(True)
                W, b = unflatten(p, unit["shapes"])
                with torch.enable_grad():
                    y = unit_forward(unit["kind"], W, b, xin)
                    wrt = [xin, p] if u > 0 else [p]
                    grads = torch.autograd.grad(y, wrt, grad_out[r])
                g.free(f"act{u}")
                grad_out[r] = grads[0] if u > 0 else None
                g.alloc(f"gtmp{u}", grads[-1].to(GRAD_DTYPE), "grad_temp")   # this unit's full grad
            self._release(u)
            self._on_unit_grad_ready(u)

        if self.stage == 1:   # ZeRO-1 reduce-scatters the full grad buffers once backward is done
            C.phase = "backward"
            for u in range(len(units)):
                shards = C.reduce_scatter([g.get(f"g{u}") for g in gpus], tag=f"grads:{units[u]['name']}")
                S = self.shard_sizes[u]
                for r, g in enumerate(gpus):
                    g.get(f"g{u}")[r * S:(r + 1) * S].copy_(shards[r])

        # ---------------- optimizer step ----------------
        C.phase = "step"
        for u in range(len(units)):
            S = self.shard_sizes[u]
            for r, g in enumerate(gpus):
                grad = g.get(f"g{u}")
                if self.stage == 1:
                    grad = grad[r * S:(r + 1) * S]        # only my slice is averaged / needed
                adamw_(g.get(f"master{u}"), g.get(f"m{u}"), g.get(f"v{u}"),
                       grad.to(MASTER_DTYPE), self.step_num, self.lr)
                if self.stage in (0, 3):                   # my bf16 params come straight from my master copy
                    g.get(f"p{u}").copy_(g.get(f"master{u}").to(PARAM_DTYPE))
            if self.stage in (1, 2):                       # everyone updated a different slice -> all-gather
                fulls = C.all_gather([g.get(f"master{u}").to(PARAM_DTYPE) for g in gpus],
                                     tag=f"params:{units[u]['name']}")
                for g, t in zip(gpus, fulls):
                    g.get(f"p{u}").copy_(t)
            for g in gpus:
                g.get(f"g{u}").zero_()
        C.phase = ""
        return sum(losses) / N

    def _on_unit_grad_ready(self, u):
        gpus = self.C.gpus
        name = self.spec[u]["name"]
        if self.stage == 0:        # DDP: all-reduce the bucket as soon as it is ready
            red = self.C.all_reduce([g.get(f"gtmp{u}") for g in gpus], tag=f"grads:{name}")
            for g, t in zip(gpus, red):
                g.get(f"g{u}").copy_(t)
        elif self.stage == 1:      # ZeRO-1: keep the full local grad, reduce later
            for g in gpus:
                g.get(f"g{u}").copy_(g.get(f"gtmp{u}"))
        else:                      # ZeRO-2/3: reduce-scatter now, keep only my shard
            shards = self.C.reduce_scatter([g.get(f"gtmp{u}") for g in gpus], tag=f"grads:{name}")
            for g, t in zip(gpus, shards):
                g.get(f"g{u}").copy_(t)
        for g in gpus:
            g.free(f"gtmp{u}")

    # --- inspection ------------------------------------------------------------------------
    def full_master_params(self):
        """Stitch the fp32 master weights back together from all ranks (for equivalence checks)."""
        gpus, out = self.C.gpus, []
        for u in range(len(self.spec)):
            if self.stage == 0:
                out.append(gpus[0].get(f"master{u}").clone())
            else:
                out.append(torch.cat([g.get(f"master{u}") for g in gpus]))
        return torch.cat(out)

    def ranks_in_sync(self):
        """In stages 0-2 every rank must hold identical bf16 params after the step."""
        if self.stage == 3:
            return True
        ref = self.C.gpus[0]
        return all(torch.equal(ref.get(f"p{u}"), g.get(f"p{u}"))
                   for g in self.C.gpus[1:] for u in range(len(self.spec)))


# ----------------------------------------------------------------------------------------
# 5. The ZeRO paper's formulas (Rajbhandari et al., 2020, Section 5 / Table 1)
# ----------------------------------------------------------------------------------------
def zero_model_state_bytes(psi, N, stage, K=12):
    """Per-GPU model-state memory in bytes (2 bytes params + 2 bytes grads + K bytes optimizer)."""
    if stage == 0:
        return (2 + 2 + K) * psi
    if stage == 1:
        return 2 * psi + 2 * psi + K * psi / N
    if stage == 2:
        return 2 * psi + (2 + K) * psi / N
    return (2 + 2 + K) * psi / N


def zero_comm_psi(stage):
    """Communication volume per step per rank, in units of Psi (paper Section 7)."""
    return {0: 2.0, 1: 2.0, 2: 2.0, 3: 3.0}[stage]


def max_params_that_fit(gpu_bytes, N, stage, K=12):
    return gpu_bytes / zero_model_state_bytes(1, N, stage, K)

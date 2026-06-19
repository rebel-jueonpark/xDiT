"""End-to-end validation of rebel-compiler all_to_all_x on 2 (or N) ATOM NPUs.

Proves the full path:
    torch.ops.rbln_custom_ops.ccl_all2all_x_kernel  (torch.library custom op)
      -> injected converter (a2a_converter_shim)
      -> relay all_to_all_x  (MLIR, already built into librbln.so on this branch)
      -> rcclAllToAllX       (librbln-ccl 3.2.2 runtime)
      -> correct permuted result.

Launch:
    cd /home/jueonpark/xdit/rbln_scratch
    torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29555 \
        test_all2all_x.py

Semantics under test (classic symmetric all-to-all):
    send_buffer is (R, t, H); send_buffer[d] is the block this rank sends to rank d.
    recv_buffer is (R, t, H); recv_buffer[s] is the block received from rank s.
    We encode send_buffer[d] := rank*10 + d, so after the exchange
    recv_buffer[s] must equal s*10 + rank on every rank. Checked numerically.
"""
import os
import sys

# --- RBLN distributed bootstrap (must precede torch_rbln import) ---
# export-mem is required for the collective to actually transfer (without it the
# collective hangs). It serializes every model I/O dtype, and the runtime enum
# has no 'uint16' — so send_sizes must NOT be a uint16 I/O; we feed fp32 and cast
# to a uint16 *intermediate* in-graph (see a2a_cast_u16). All model I/O is bf16/fp32.
os.environ.setdefault("RCCL_FORCE_EXPORT_MEM", "1")
os.environ.setdefault("RCCL_PORT_GEN", "1")
os.environ.setdefault("RBLN_FORCE_CCL_ASYNC", "1")  # vllm-rbln default; sync mode deadlocks the collective
os.environ.setdefault("RBLN_ROOT_IP", "127.0.0.1")
os.environ.setdefault("RBLN_LOCAL_IP", "127.0.0.1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29555")

import torch
import torch.distributed as dist
import torch_rbln  # noqa: F401  registers torch.rbln + rbln-ccl backend

# Inject the all_to_all_x converter into rebel BEFORE any compile.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a2a_converter_shim  # noqa: F401

from torch import Tensor
from torch.rbln import set_device, device_count


# --- The custom op (declaration mirrors vllm-rbln all2all.py:105-133) ---
@torch.library.custom_op("rbln_custom_ops::ccl_all2all_x_kernel", mutates_args=())
def ccl_all2all_x_kernel(
    send_buffer: Tensor, send_sizes: Tensor, ccl_world_size: int, group_name: str
) -> Tensor:
    # CPU stub (only used in eager / fake tracing; under backend=rbln the
    # converter replaces this with the real collective).
    R = ccl_world_size
    t = send_buffer.shape[1]
    H = send_buffer.shape[2]
    return torch.zeros(R, t, H, dtype=send_buffer.dtype)


@ccl_all2all_x_kernel.register_fake
def _ccl_all2all_x_kernel_fake(send_buffer, send_sizes, ccl_world_size, group_name):
    R = ccl_world_size
    t = send_buffer.shape[1]
    H = send_buffer.shape[2]
    return torch.empty(R, t, H, dtype=send_buffer.dtype)


# Cast helper: produces a uint16 in-graph intermediate (slot) via a relay-level
# cast, so send_sizes is never a uint16 model I/O (the runtime export enum lacks
# uint16) and the compile stays on the export path (torch's .to(uint16) does not).
@torch.library.custom_op("rbln_custom_ops::a2a_cast_u16", mutates_args=())
def a2a_cast_u16(x: Tensor) -> Tensor:
    return x.to(torch.uint16)


@a2a_cast_u16.register_fake
def _a2a_cast_u16_fake(x):
    return torch.empty_like(x, dtype=torch.uint16)


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    dist.init_process_group(
        backend="rbln-ccl", init_method="env://", world_size=world, rank=rank
    )
    set_device(rank % device_count())
    dev = f"rbln:{local_rank}"

    R = world
    t, H = 64, 64  # t multiple of 64; H 16-bit-aligned inner dim

    # The rebel runtime builds CCL subgroups itself by splitting its default
    # (WORLD) comm using the colors in process_group_dict. Creating a *torch*
    # rbln-ccl new_group here would do a SECOND rcclCommSplit of the same comm
    # and deadlocks. For a 2-rank test the WORLD comm already has the right
    # membership, so we use it directly: op group="-1" makes ResolveComms() fall
    # back to default_comms (the WORLD comm), and process_group_dict stays empty
    # (no runtime split). A gloo group is kept only for the host-side barrier.
    ranks = list(range(world))
    op_group = "-1"
    process_group_dict = {}
    cpu_group = dist.new_group(ranks, backend="gloo")

    if rank == 0:
        print(f"[rank0] world={world} op_group={op_group} dev={dev}", flush=True)

    # send_buffer[d] := rank*10 + d. The module adds +1 (a compute op) before the
    # collective, so recv_buffer[s] must == s*10 + rank + 1 on every rank.
    send = torch.empty(R, t, H, dtype=torch.bfloat16, device=dev)
    for d in range(R):
        send[d].fill_(float(rank * 10 + d))
    # send_sizes fed as fp32 (export-serializable), cast to a uint16 intermediate
    # in-graph via a2a_cast_u16. Both CCL-op inputs end up as compute-op outputs
    # (RBLN slots); a raw input feeding the CCL op is rejected ("CCL op should not
    # have input_desc"), and a uint16 model I/O breaks memory export.
    send_sizes_f32 = torch.zeros(R, 64, dtype=torch.float32, device=dev)
    send_sizes_f32[:, 0] = float(t)

    class A2A(torch.nn.Module):
        def forward(self, send_buffer, send_sizes_f32):
            x = send_buffer + 1.0
            # fp32 input -> cast to uint16 (intermediate, so uint16 is never a
            # model I/O that memory-export can't serialize) -> cat re-materializes
            # it as a genuine uint16 device slot and launders the input_desc chain
            # (a bare cast yields a non-slot; cat of a uint16 tensor yields a slot).
            ss_u = torch.ops.rbln_custom_ops.a2a_cast_u16(send_sizes_f32)
            ss = torch.cat([ss_u, ss_u], dim=0)[:R]
            return torch.ops.rbln_custom_ops.ccl_all2all_x_kernel(
                x, ss, R, op_group
            )

    from rebel.core.torch_compile import rbln_backend
    try:
        from rebel import CompileContext
    except ImportError:
        from rebel.compile_context import CompileContext

    opts = {
        "tensor_parallel_size": world,
        "process_group_dict": process_group_dict,
        "mode": ["strict"],
        "compile_context": CompileContext(),
        "guard_filter_fn": torch.compiler.keep_tensor_guards_unsafe,
        "cache_dir": os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"a2a_cache_rank{rank}"
        ),
    }

    module = A2A().to(dev)
    compiled = torch.compile(module, backend=rbln_backend, options=opts, dynamic=False)
    recv = compiled(send, send_sizes_f32)
    recv_cpu = recv.to("cpu").float()

    ok = True
    for s in range(R):
        expected = float(s * 10 + rank + 1)
        got = recv_cpu[s].mean().item()
        mn, mx = recv_cpu[s].min().item(), recv_cpu[s].max().item()
        good = abs(got - expected) < 0.5 and abs(mn - mx) < 0.5
        ok = ok and good
        print(
            f"[rank{rank}] recv[{s}] mean={got:.2f} (min={mn:.2f} max={mx:.2f}) "
            f"expected={expected:.0f} {'OK' if good else 'BAD'}",
            flush=True,
        )

    print(f"[rank{rank}] RESULT: {'PASS' if ok else 'FAIL'}", flush=True)

    dist.barrier(group=cpu_group)
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

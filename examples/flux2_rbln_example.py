"""Run FLUX.2 (klein-4B / klein-9B / dev) on Rebellions NPU via xDiT.

Launch via torchrun, e.g.:

  torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29501 \
      examples/flux2_rbln_example.py \
      --model FLUX.2-klein-4B \
      --ulysses_degree 4 \
      --height 512 --width 512 --num_inference_steps 4 \
      --prompt "a futuristic city at dusk"
"""

import os
import sys

# RBLN distributed bootstrap env (must be set before torch_rbln C++ libs load).
os.environ.setdefault("RCCL_FORCE_EXPORT_MEM", "1")
os.environ.setdefault("RCCL_PORT_GEN", "1")  # multi-NPU rbln-ccl port generation
os.environ.setdefault("RBLN_ROOT_IP", "127.0.0.1")
os.environ.setdefault("RBLN_LOCAL_IP", "127.0.0.1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

# Point at the shared HF cache so we pick up the auth token and any cached weights.
os.environ.setdefault(
    "HF_HOME", "/mnt/shared_data/groups/sw_dev/.cache/huggingface"
)

import torch  # noqa: E402
import torch_rbln  # noqa: F401,E402  (registers torch.rbln + rbln-ccl backend)

from xfuser.config.args import FlexibleArgumentParser, xFuserArgs  # noqa: E402
from xfuser.runner import xFuserModelRunner, setup_logging  # noqa: E402
from xfuser.core.utils.runner_utils import log  # noqa: E402


def main() -> int:
    setup_logging()
    parser = FlexibleArgumentParser(description="FLUX.2 on Rebellions NPU via xDiT")
    xfuser_args = xFuserArgs.add_runner_args(parser).parse_args()
    args = vars(xfuser_args)

    runner = xFuserModelRunner(args)
    runner.print_args(args)

    input_args = runner.preprocess_args(args)
    runner.initialize(input_args)
    output, timings = runner.run(input_args)
    runner.save(output=output, timings=timings)
    runner.cleanup()
    log("FLUX.2 RBLN run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

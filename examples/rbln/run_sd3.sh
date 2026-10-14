#!/usr/bin/env bash
# Run SD3-medium on Rebellions ATOM NPUs through the xDiT runner.
#
#   NPROC=1 examples/rbln/run_sd3.sh                       # one NPU, compiled
#   NPROC=2 examples/rbln/run_sd3.sh --use_cfg_parallel    # CFG parallel on two NPUs
#   NPROC=2 examples/rbln/run_sd3.sh --ulysses_degree 2    # Ulysses sequence parallel
#   NPROC=4 examples/rbln/run_sd3.sh --use_cfg_parallel --ulysses_degree 2
#
# Any extra argument is passed to xfuser.runner. RBLN_DEVICES selects the NPUs
# (torch-rbln's CUDA_VISIBLE_DEVICES); RBLN_NPUS_PER_DEVICE=k makes every rank's logical
# device k physical NPUs, which the rebel compiler then shards itself (RSD).
# TEXT_ENCODER_DEVICE=spare puts the text encoders (T5-XXL, CLIP) on one NPU beyond the
# ranks' own instead of the host; it needs NPROC+1 visible NPUs (set below unless given).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"

NPROC=${NPROC:-1}
# torchrun sets OMP_NUM_THREADS=1 for multi-process launches; the host-side text encoders
# (T5-XXL) need real threads. Split the cores between the ranks unless the caller chose.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( $(nproc) / NPROC ))}
MODEL=${MODEL:-SD3-medium}
OUT=${OUT:-./outputs/rbln}
TEXT_ENCODER_DEVICE=${TEXT_ENCODER_DEVICE:-cpu}
if [ "${TEXT_ENCODER_DEVICE}" = spare ] && [ -z "${RBLN_DEVICES:-}" ]; then
  export RBLN_DEVICES=$(seq -s, 0 "${NPROC}")
fi
PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

torchrun --nnodes=1 --nproc_per_node="${NPROC}" --master_port="${PORT}" \
  -m xfuser.runner \
  --model "${MODEL}" \
  --use_torch_compile --torch_compile_backend rbln \
  --rbln_text_encoder_device "${TEXT_ENCODER_DEVICE}" \
  --height "${HEIGHT:-1024}" --width "${WIDTH:-1024}" \
  --num_inference_steps "${STEPS:-28}" \
  --prompt "${PROMPT:-A photo of a cat wearing a tiny wizard hat, studio lighting}" \
  --output_directory "${OUT}" \
  --num_iterations "${ITER:-2}" \
  "$@"

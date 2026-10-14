# Environment for xDiT on Rebellions ATOM NPUs (source this before torchrun).
#
# torch-rbln registers the `rbln` device and the `rbln-ccl` torch.distributed backend;
# the rebel compiler serves torch.compile(backend="rbln"). These variables must be set
# in every rank before torch_rbln is imported, so the launcher exports them and
# torchrun inherits them.

# Control plane for rbln-ccl bootstrap on a single host.
export RBLN_ROOT_IP=${RBLN_ROOT_IP:-127.0.0.1}
export RBLN_LOCAL_IP=${RBLN_LOCAL_IP:-127.0.0.1}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
# Export device memory to peers for collectives; async CCL issue (vllm-rbln defaults).
export RCCL_FORCE_EXPORT_MEM=${RCCL_FORCE_EXPORT_MEM:-1}
export RBLN_FORCE_CCL_ASYNC=${RBLN_FORCE_CCL_ASYNC:-1}
# rbln-ccl's unique-id / autoport init path: lets several multi-rank jobs share a host
# (each job's torchrun MASTER_PORT must differ; the launcher randomises it).
export RCCL_PORT_GEN=${RCCL_PORT_GEN:-1}
# SD3 weights live in the shared HF cache on this host; stay offline for gated repos.
export HF_HOME=${HF_HOME:-/mnt/shared_data/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

# A compiler *source* checkout exports REBEL_HOME / TVM_HOME and puts its build dir on
# PATH / LD_LIBRARY_PATH; the rebel-compiler wheel honours TVM_HOME and would then load
# that tree's (stale) librbln.so and die on an undefined symbol or a segfault. The wheel
# is self-contained, so drop all of it for this process tree.
# With an editable (source-built) rebel-compiler in the venv those variables are the
# point, so XDIT_RBLN_SOURCE_COMPILER=1 keeps them.
if [ "${XDIT_RBLN_SOURCE_COMPILER:-0}" != "1" ]; then
  unset REBEL_HOME TVM_HOME TVM_LIBRARY_PATH REBEL_CONFIG_PATH
  export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v 'rebel_compiler/build' | paste -sd:)
  export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'rebel_compiler' | paste -sd:)
fi

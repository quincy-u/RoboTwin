#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
SIMPLE_GRASP_ROOT=${SIMPLE_GRASP_ROOT:-"$HOME/projects/simple-grasp"}
PYTHON="$REPO_ROOT/.venv/bin/python"
UV=${UV:-uv}
HYDRA_VERSION=1.3.2
OMEGACONF_VERSION=2.3.0
M2T2_CHECKPOINT_REVISION=a20325dcd19cd1b838274974d0ebe35bfe383796
M2T2_CHECKPOINT_SHA256=e35c3cb11e06f46c5d406bdfc756bc06f48b256dd6d638408d9a5ff13deb97fb
CHECKPOINT_DIR="$SIMPLE_GRASP_ROOT/checkpoints"
CHECKPOINT_PATH="$CHECKPOINT_DIR/m2t2.pth"

if [[ ! -x "$PYTHON" ]]; then
    echo "RoboTwin environment not found at $PYTHON" >&2
    exit 1
fi
if [[ ! -d "$SIMPLE_GRASP_ROOT/third_party/M2T2" ]]; then
    echo "simple-grasp checkout not found at $SIMPLE_GRASP_ROOT" >&2
    exit 1
fi

"$UV" pip install --python "$PYTHON" \
    "hydra-core==$HYDRA_VERSION" \
    "omegaconf==$OMEGACONF_VERSION"
"$UV" pip install --python "$PYTHON" "$SIMPLE_GRASP_ROOT"
CUDA_HOME=${CUDA_HOME:-/software/cuda-13.0} \
    "$UV" pip install --python "$PYTHON" \
    "$SIMPLE_GRASP_ROOT/third_party/M2T2/pointnet2_ops" \
    --no-build-isolation

mkdir -p "$CHECKPOINT_DIR"
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    checkpoint_tmp=$(mktemp "$CHECKPOINT_DIR/.m2t2.pth.XXXXXX")
    cleanup_checkpoint_tmp() {
        if [[ -n "${checkpoint_tmp:-}" ]]; then
            rm -f -- "$checkpoint_tmp"
        fi
    }
    trap cleanup_checkpoint_tmp EXIT
    curl -L --fail --progress-bar \
        "https://huggingface.co/wentao-yuan/m2t2/resolve/$M2T2_CHECKPOINT_REVISION/m2t2.pth" \
        -o "$checkpoint_tmp"
    printf "%s  %s\n" "$M2T2_CHECKPOINT_SHA256" "$checkpoint_tmp" | sha256sum --check --status
    mv -- "$checkpoint_tmp" "$CHECKPOINT_PATH"
    checkpoint_tmp=
    trap - EXIT
fi

printf "%s  %s\n" "$M2T2_CHECKPOINT_SHA256" "$CHECKPOINT_PATH" | sha256sum --check --status || {
    echo "M2T2 checkpoint checksum mismatch: $CHECKPOINT_PATH" >&2
    exit 1
}

"$PYTHON" -c "import hydra, pointnet2_ops, torch; print('M2T2 dependencies ready:', torch.cuda.is_available())"

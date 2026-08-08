#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
VENV_DIR=${VIRTUAL_ENV:-"$REPO_ROOT/.venv"}
PYTHON="$VENV_DIR/bin/python"

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not installed or is not available on PATH." >&2
    exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        echo "Error: the active virtual environment has no Python executable: $PYTHON" >&2
        exit 1
    fi
    echo "Creating a Python 3.10 environment at $VENV_DIR ..."
    uv venv --python 3.10 "$VENV_DIR"
fi

uv_pip() {
    local command=$1
    shift
    uv pip "$command" --python "$PYTHON" "$@"
}

echo "Using $($PYTHON --version) from $PYTHON"

echo "Installing build tooling ..."
# SAPIEN 3.0.0b1 imports pkg_resources, which was removed in setuptools 81.
uv_pip install "setuptools<81" wheel ninja

echo "Installing the necessary packages ..."
uv_pip install -r "$SCRIPT_DIR/requirements.txt"

if [[ -z "${CUDA_HOME:-}" ]]; then
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_HOME=$(cd -- "$(dirname -- "$(command -v nvcc)")/.." && pwd)
    else
        TORCH_CUDA_VERSION=$($PYTHON -c 'import torch; print(torch.version.cuda or "")')
        CUDA_CANDIDATE="/software/cuda-$TORCH_CUDA_VERSION"
        if [[ -n "$TORCH_CUDA_VERSION" && -x "$CUDA_CANDIDATE/bin/nvcc" ]]; then
            CUDA_HOME="$CUDA_CANDIDATE"
        else
            echo "Error: CUDA toolkit not found. Set CUDA_HOME to its install directory." >&2
            exit 1
        fi
    fi
    export CUDA_HOME
fi
export PATH="$CUDA_HOME/bin:$PATH"
echo "Using CUDA toolkit from $CUDA_HOME"

echo "Installing pytorch3d ..."
uv_pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

echo "Adjusting code in sapien/wrapper/urdf_loader.py ..."
# location of sapien, like "~/.conda/envs/RoboTwin/lib/python3.10/site-packages/sapien"
SAPIEN_LOCATION=$($PYTHON -c 'import pathlib, sapien; print(pathlib.Path(sapien.__file__).resolve().parent)')
URDF_LOADER=$SAPIEN_LOCATION/wrapper/urdf_loader.py
# ----------- before -----------
# 667         with open(urdf_file, "r") as f:
# 668             urdf_string = f.read()
# 669 
# 670         if srdf_file is None:
# 671             srdf_file = urdf_file[:-4] + "srdf"
# 672         if os.path.isfile(srdf_file):
# 673             with open(srdf_file, "r") as f:
# 674                 self.ignore_pairs = self.parse_srdf(f.read())
# ----------- after  -----------
# 667         with open(urdf_file, "r", encoding="utf-8") as f:
# 668             urdf_string = f.read()
# 669 
# 670         if srdf_file is None:
# 671             srdf_file = urdf_file[:-4] + ".srdf"
# 672         if os.path.isfile(srdf_file):
# 673             with open(srdf_file, "r", encoding="utf-8") as f:
# 674                 self.ignore_pairs = self.parse_srdf(f.read())
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$URDF_LOADER"

echo "Adjusting code in mplib/planner.py ..."
# location of mplib, like "~/.conda/envs/RoboTwin/lib/python3.10/site-packages/mplib"
MPLIB_LOCATION=$($PYTHON -c 'import pathlib, mplib; print(pathlib.Path(mplib.__file__).resolve().parent)')
# Adjust some code in planner.py
# ----------- before -----------
# 807             if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
# 808                 return {"status": "screw plan failed"}
# ----------- after  ----------- 
# 807             if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
# 808                 return {"status": "screw plan failed"}
PLANNER=$MPLIB_LOCATION/planner.py
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$PLANNER"

echo "Installing Curobo ..."
CUROBO_DIR="$REPO_ROOT/envs/curobo"
if [[ ! -d "$CUROBO_DIR/.git" ]]; then
    git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$CUROBO_DIR"
else
    echo "Using existing Curobo checkout at $CUROBO_DIR"
fi
# Curobo v0.7.8 defines unused global lerp overloads that conflict with
# std::lerp when CUDA 13 compiles the extensions as C++20.
CUROBO_HELPER_MATH="$CUROBO_DIR/src/curobo/curobolib/cpp/helper_math.h"
sed -i -E '/^inline __device__ __host__ float[234]? lerp/s/ lerp/ curobo_lerp/' "$CUROBO_HELPER_MATH"

uv_pip install -e "$CUROBO_DIR" --no-build-isolation

echo "Installation basic environment complete!"
echo -e "You need to:"
echo -e "    1. \033[34m\033[1m(Important!)\033[0m Download assets from huggingface."
echo -e "    2. Install requirements for running baselines. (Optional)"
echo "See INSTALLATION.md for more instructions."

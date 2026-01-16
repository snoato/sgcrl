#!/bin/bash
#SBATCH --job-name=stretch_cpc
#SBATCH --partition=rleap_gpu_24gb
#SBATCH --account=rleap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_%A.out
#SBATCH --error=logs/slurm_%A.err

# Initialize conda and activate environment
source /work/rleap1/jaxon.cheng/miniforge3/etc/profile.d/conda.sh
conda activate /work/rleap1/jaxon.cheng/venvs/contrastive_rl

# Set environment variables
export LD_LIBRARY_PATH=/work/rleap1/jaxon.cheng/venvs/contrastive_rl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/rleap1/jaxon.cheng/mujoco/mujoco210/mujoco210/bin:$LD_LIBRARY_PATH
export MUJOCO_PY_MUJOCO_PATH=/work/rleap1/jaxon.cheng/mujoco/mujoco210/mujoco210
export MUJOCO_GL=osmesa
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CUDA_HOME=/usr/local/cuda-11.7
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"

echo "CUDA_HOME=$CUDA_HOME"
which ptxas || (echo "ptxas not found on PATH" && exit 1)
ptxas --version || true

# Increase file descriptor limit
ulimit -n 4096

# =====================================================
# 🔍 CUDA / JAX DIAGNOSTICS 
# =====================================================
echo "===== GPU INFO ====="
nvidia-smi || true

echo "===== CUDA TOOLCHAIN ====="
which ptxas || echo "ptxas NOT FOUND"
which nvcc  || echo "nvcc NOT FOUND"

echo "===== CUDA DIR CHECK ====="
ls -l /usr/local/cuda/bin/ptxas 2>/dev/null || echo "/usr/local/cuda/bin/ptxas not found"

echo "===== PATH ====="
echo $PATH
# =====================================================

# Change to working directory
cd /work/rleap1/jaxon.cheng/sgcrl

# Run training
python -c "import contrastive.utils; print('contrastive.utils =', contrastive.utils.__file__)"
python lp_contrastive.py \
        --env=stretch_pick \
        --alg=contrastive_cpc \
        --num_steps=6000 \
        --seed=42 \
        --add_uid=True \

echo "Training completed!"

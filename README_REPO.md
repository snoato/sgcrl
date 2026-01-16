Single-goal Contrastive RL (SGCRL)
better Setup

Install package dependencies:
1. Change library path: export LD_LIBRARY_PATH=/work/rleap1/jaxon.cheng/venvs/contrastive_rl/lib:$LD_LIBRARY_PATH

2. Install the requirements: pip install -r requirements.txt --no-deps

3. Download the mujoco binaries and place them in ~/.mujoco/ according to instructions in https://github.com/openai/mujoco-py. Run export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/rleap1/jaxon.cheng/mujoco/mujoco210/mujoco210/bin:$LD_LIBRARY_PATH

export MUJOCO_PY_MUJOCO_PATH=/work/rleap1/jaxon.cheng/mujoco/mujoco210/mujoco210
export MUJOCO_GL=osmesa or export MUJOCO_GL=glfw
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia


4. Reinstall strict versions for the following packages:
    pip install dm-acme[jax,tf] cd sgcrl

    pip install jax==0.4.10 jaxlib==0.4.10
    pip install ml_dtypes==0.2.0
    pip install dm-haiku==0.0.9
    pip install gymnasium-robotics 
    pip uninstall scipy; pip install scipy==1.12
    pip install torch==2.1.2 scikit-learn pandas

    pip install protobuf==3.19.1
    pip install optax==0.1.2
    pip install jax==0.4.10 jaxlib==0.4.10
    pip install cython==0.29.30

    conda install -c conda-forge glew -y
    pip install patchelf

    #For GPU: given nvidia-smi says CUDA version 12.1(desk-01) or 12.2 (server) -> using CUDA toolkit 11.8:
    pip install optax==0.1.7
    pip install --upgrade jax==0.4.7 jaxlib==0.4.7+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.6/lib64

    For information see:

    https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    https://pytorch.org/get-started/previous-versions/
    https://docs.nvidia.com/deploy/cuda-compatibility/

    #clean
    pip uninstall -y \
    nvidia-cublas-cu12 \
    nvidia-cuda-cupti-cu12 \
    nvidia-cuda-nvrtc-cu12 \
    nvidia-cuda-runtime-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-cufft-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-nccl-cu12 \
    nvidia-nvjitlink-cu12 \
    nvidia-nvtx-cu12


    pip install optax==0.1.7
    pip install --upgrade jax==0.4.7 jaxlib==0.4.7+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


    conda install -y -c nvidia -c conda-forge \
    cudatoolkit=11.8 \
    cudnn=8.2.0
    conda install mamba -c conda-forge #chatgpt says thats faster at solving
    mamba install pytorch==2.1.2 pytorch-cuda=11.8 cudnn=8.2.0 -c pytorch -c nvidia -c conda-forge

    mamba install conda-forge::glfw
    mamba install anaconda::pyopengl

    mamba install mesalib -c conda-forge
    mamba install libgl -c conda-forge
    export MUJOCO_GL=osmesa

    mamba install -c conda-forge glew
    mamba install -c conda-forge mesalib
    mamba install -c menpo glfw3
    mamba install conda-forge::libopengl

    export PATH=/usr/local/cuda-12.1/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

5. Evrytime activate please type in:
    
    export LD_LIBRARY_PATH=/work/rleap1/jaxon.cheng/venvs/contrastive_rl/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/rleap1/jaxon.cheng/mujoco/mujoco210/mujoco210/bin:$LD_LIBRARY_PATH
    export MUJOCO_PY_MUJOCO_PATH=/work/rleap1/jaxon.cheng/mujoco/mujoco210/mujoco210
    export MUJOCO_GL=osmesa or export MUJOCO_GL=glfw
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

   if want to change an environment:
    python lp_contrastive.py --env=sawyer_box
    python lp_contrastive.py --env=sawyer_peg

6. 
    cd sgcrl

    python lp_contrastive.py \
    --env=stretch_pick \
    --alg=contrastive_cpc \
    --render \
    --num_steps=100000 \
    --seed=42 \
    --add_uid=True \


 




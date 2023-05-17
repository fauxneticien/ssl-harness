# NVIDIA official container with CUDA 11.7 (note pytorch-cuda=11.7 in PyTorch install below)
# See https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
FROM nvcr.io/nvidia/pytorch:22.05-py3

# Update container PyTorch from 1.9 to 1.13.1
RUN conda install pytorch==1.13.1 \
    torchvision==0.14.1 \
    torchaudio==0.13.1 \
    pytorch-cuda=11.7 \
    -c pytorch \
    -c nvidia

# Re-build Apex for newer version of PyTorch
WORKDIR /apex

RUN git clone https://github.com/NVIDIA/apex /apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
    --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
    --global-option="--fast_multihead_attn" ./

RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt update \
    && apt install -y \
    gh \
    rclone

RUN pip install pytorch-lightning==1.8.6

WORKDIR /workspace

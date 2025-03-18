# This assumes the container is running on a system with a CUDA GPU
#ähttps://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-25-01.html
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Set DEBIAN_FRONTEND temporarily for build-time only
ARG DEBIAN_FRONTEND=noninteractive

# ARG USER_UID
# ARG USER_GID
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Set environment variable to enable dynamic GPU memory allocation
ENV TF_CPP_MIN_LOG_LEVEL 3
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV CUDA_VISIBLE_DEVICES=0
ENV HOME=/home/$USERNAME


RUN apt-get update -y && apt-get upgrade -y  \
    && apt-get install -y sudo wget curl unzip git ffmpeg libsm6 libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# UV setup and sync
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

USER $USERNAME
WORKDIR $HOME

RUN pip install tensorflow==2.18.0 hydra-core ipykernel ipywidgets tqdm seaborn opendatasets mlflow optuna optuna-integration[tfkeras]

# ENV VIRTUAL_ENV="$HOME/.venv" 
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# COPY pyproject.toml .
# RUN uv sync --active 



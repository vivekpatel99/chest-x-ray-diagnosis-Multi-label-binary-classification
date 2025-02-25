# This assumes the container is running on a system with a CUDA GPU

FROM nvcr.io/nvidia/tensorflow:25.01-tf2-py3

# Set DEBIAN_FRONTEND temporarily for build-time only
ARG DEBIAN_FRONTEND=noninteractive

# ARG USER_UID
# ARG USER_GID
ARG USERNAME=builder
ARG USER_UID=1001
ARG USER_GID=$USER_UID


ENV HOME=/home/$USERNAME

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -d $HOME $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME   \
    && sudo chown -R $USERNAME:$USERNAME $HOME

RUN apt-get update -y && apt-get upgrade -y  \
    && apt-get install -y wget curl unzip git ffmpeg libsm6 libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# UV setup and sync
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

USER $USERNAME
WORKDIR $HOME

ENV VIRTUAL_ENV="$HOME/.venv" 
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# COPY pyproject.toml .
# RUN uv sync --active 

# Set environment variable to enable dynamic GPU memory allocation
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV CUDA_VISIBLE_DEVICES=0
ENV TF_CPP_MIN_LOG_LEVEL 3


ENV PATH="/root/.local/bin:${PATH}"


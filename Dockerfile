# Base Image: Ubuntu 24.04
FROM ubuntu:24.04

# Install essential build tools + perf (generic version)
RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    cmake \
    ninja-build \
    git \
    python3 \
    linux-tools-common \
    linux-tools-generic && \
    rm -rf /var/lib/apt/lists/*

# This will be your project root *inside* the container.
# We won't COPY the source in the image – we'll MOUNT it at runtime.
WORKDIR /src

# Default command: open a shell in the container
CMD ["/bin/bash"]

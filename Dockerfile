# Base Image: Ubuntu 22.04 LTS
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

# Create working directory inside the container
WORKDIR /spira

# Copy entire project into container
COPY . . 

# Configure project (Release)
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build the project (Release mode)
RUN cmake --build build -j

# Default command: open a shell in the container
CMD ["/bin/bash"]

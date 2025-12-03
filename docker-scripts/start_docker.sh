echo "Starting Docker"
docker build -t spira-env .
docker run -it --privileged --cpuset-cpus="0" spira-env


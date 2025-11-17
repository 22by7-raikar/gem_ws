FROM ubuntu:20.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    cmake \
    python3-pip \
    python3-dev \
    x11-apps \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install ROS Noetic
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - && \
    echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list && \
    apt-get update && apt-get install -y \
    ros-noetic-desktop-full \
    ros-noetic-catkin \
    ros-noetic-tf-transformations \
    python3-catkin-pkg-modules \
    python3-rospkg-modules \
    && rm -rf /var/lib/apt/lists/*

# Install Gazebo and GEM simulator
RUN apt-get update && apt-get install -y \
    ros-noetic-gazebo-ros \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-plugins \
    ros-noetic-ackermann-msgs \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install \
    torch==1.13.1 \
    numpy>=1.19 \
    pandas>=1.1 \
    matplotlib>=3.3 \
    cvxpy>=1.5 \
    scipy>=1.5

# Setup ROS environment
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Create workspace directory
RUN mkdir -p /root/gem_ws/src

# Set working directory
WORKDIR /root/gem_ws

# Copy workspace contents
COPY . /root/gem_ws/

# Initialize and update submodules
RUN cd /root/gem_ws && git init && git submodule update --init --recursive || echo "Submodule init skipped"

# Build workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Add workspace setup to bashrc
RUN echo "source /root/gem_ws/devel/setup.bash" >> ~/.bashrc

# Run bash
CMD ["/bin/bash"]

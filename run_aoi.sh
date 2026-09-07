#!/usr/bin/env bash
# Startup script for AOI PCB Defect Detection System

# Source ROS 2 Foxy
if [ -f /opt/ros/foxy/setup.bash ]; then
    source /opt/ros/foxy/setup.bash
fi

# Fix Jetson static TLS block allocation for GStreamer & PyTorch
if [ -f /lib/aarch64-linux-gnu/libGLdispatch.so.0 ]; then
    export LD_PRELOAD="/lib/aarch64-linux-gnu/libGLdispatch.so.0:$LD_PRELOAD"
fi

cd "$(dirname "$0")"
echo "Starting AOI PCB Inspection System with YOLOv12 and ROS 2..."
python3 app.py "$@"

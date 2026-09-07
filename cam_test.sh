#!/usr/bin/env bash

OUT="jetson_camera_diagnose_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee "$OUT") 2>&1

echo "========================================="
echo "      Jetson Camera Diagnose Tool"
echo "========================================="
echo

echo "===== System ====="
uname -a
echo
cat /etc/nv_tegra_release
echo
dpkg -l | egrep "nvidia-l4t|libnvinfer|python3-libnvinfer"

echo
echo "===== Camera Device ====="
ls -l /dev/video*
ls -l /dev/v4l-subdev*

echo
echo "===== Media Graph ====="
media-ctl -p

echo
echo "===== V4L2 Formats ====="
v4l2-ctl --list-formats-ext -d /dev/video0

echo
echo "===== V4L2 Streaming ====="
timeout 5 \
v4l2-ctl \
-d /dev/video0 \
--set-fmt-video=width=1920,height=1080,pixelformat=RG10 \
--set-ctrl bypass_mode=0 \
--set-ctrl sensor_mode=0 \
--stream-mmap \
--stream-count=60

echo
echo "===== Argus Daemon ====="
systemctl status nvargus-daemon --no-pager

echo
echo "===== Restart Argus ====="
sudo systemctl restart nvargus-daemon
sleep 3

echo
echo "===== Argus Pipeline ====="
timeout 10 \
gst-launch-1.0 \
nvarguscamerasrc sensor-id=0 num-buffers=30 ! \
'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=60/1' ! \
fakesink

echo
echo "===== GStreamer Plugin ====="
gst-inspect-1.0 nvarguscamerasrc

echo
echo "===== Device Tree ====="
find /proc/device-tree \
-iname "*imx477*" \
-o -iname "*camera*" 2>/dev/null

echo
echo "===== I2C ====="
i2cdetect -l

echo
echo "===== Kernel Camera Log ====="
sudo dmesg | grep -Ei \
"imx|camera|csi|tegra|argus|nvbuf|vi|isp"

echo
echo "========================================="
echo "Finished."
echo "Log saved to:"
echo "$OUT"
echo "========================================="
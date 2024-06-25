#!/bin/bash

# Start XRDP service
sudo service xrdp start
echo "STARTING XRDP"
# Export VirtualGL environment variables
export VGL_DISPLAY=:0

# Navigate to CARLA directory
cd /home/reu/carla

# Use VirtualGL to launch CARLA
vglrun make launch
echo "test"

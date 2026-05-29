#!/bin/bash

# Script to run auto_takeoff_land node with proper DDS configuration
# This script sets up the FastRTPS profile to avoid payload size warnings

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"

# Set FastRTPS profile file path
export FASTRTPS_DEFAULT_PROFILES_FILE="$PACKAGE_DIR/fastrtps_profile.xml"

# Enable QoS from XML
export RMW_FASTRTPS_USE_QOS_FROM_XML=1

echo "Using FastRTPS profile: $FASTRTPS_DEFAULT_PROFILES_FILE"
echo "Starting auto_takeoff_land node..."

# Source ROS2 workspace if not already sourced
if [ -z "$ROS_DISTRO" ]; then
    echo "Warning: ROS2 environment not sourced. Please source your workspace first."
    echo "Run: source /home/helei/TVC_ws/install/setup.bash"
fi

# Run the node
ros2 run tvc_controller auto_takeoff_land


# PX4 Thrust Vector Control (TVC) Workspace

[![ROS 2](https://img.shields.io/badge/ROS-2-blue)](https://docs.ros.org/en/humble/index.html)
[![PX4](https://img.shields.io/badge/PX4-Compatible-green)](https://px4.io/)
[![Gazebo](https://img.shields.io/badge/Gazebo-Harmonic-orange)](https://gazebosim.org/)

This ROS 2 workspace provides a complete thrust vector control (TVC) implementation for inverted coaxial drones with PX4 autopilot integration. The workspace includes a modified PX4 autopilot, ROS 2 communication bridges, message definitions, and a custom LQR controller for precise thrust vector control of coaxial motor systems with gimbal-based thrust vectoring.

![](/assets/test_1.gif)
![](/assets/test_2.gif)

## 🏗️ Workspace Structure

```
TVC_ws/
├── README.md                    # This documentation
├── assets/                      # Media files and documentation assets
│   ├── test_1.gif              # TVC demonstration video 1
│   └── test_2.gif              # TVC demonstration video 2
├── PX4_tvc/                     # Modified PX4 autopilot for TVC
│   ├── src/                     # PX4 source code
│   ├── msg/                     # PX4 message definitions
│   ├── launch/                  # PX4 launch configurations
│   ├── boards/                  # Hardware board configurations
│   └── ...                      # Standard PX4 structure
└── src/
    ├── px4_msgs/                # PX4 message definitions for ROS 2
    ├── px4_ros_com/             # PX4-ROS 2 communication bridge
    └── tvc_controller/          # Main TVC controller package
        ├── config/
        │   └── tvc_params.yaml          # Controller parameters
        ├── launch/
        │   └── lqr.launch.py            # ROS 2 launch file
        ├── models/
        │   └── tvc/                     # Gazebo TVC drone model
        │       ├── model.sdf            # SDF model definition
        │       ├── model.config         # Model configuration
        │       └── meshes/              # 3D mesh files
        ├── shell_scripts/
        │   └── run.sh                   # Automated test script using tmux (has bugs)
        ├── test/                        # Unit tests
        └── tvc_controller/
            ├── lqr.py                   # LQR controller implementation
            └── lqr_controller_node.py   # Main controller ROS 2 node
```

## 📋 Prerequisites

### System Requirements
- **OS**: Ubuntu 22.04 LTS (recommended) or Ubuntu 24.04 LTS
- **ROS 2**: Humble Hawksbill (recommended) or Galactic Geochelone
- **Python**: 3.10.12 (recommended)
- **PX4**: v1.15.4 (SITL or hardware)
- **GZ**: Harmonic (recommended) or Ionic

### Dependencies
- ROS 2 (with `colcon` build tools)
- PX4 Autopilot with uXRCE-DDS bridge
- Python packages:
  - `numpy 1.16.0`
  - `scipy 1.15.0`

## 💿 Installation
### 1. ROS2 Humble
Install ROS2 Humble following official [documentation](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html).

### 2. GZ Harmonic 

Install GZ Harmonic using official [documentation](https://gazebosim.org/docs/harmonic/install_ubuntu/).

Install GZ ROS2 bridge using following commamnd.
```bash
sudo apt install ros-humble-ros-gzharmonic
```

### 3. MicroDDS(uXRCE-DDS) 
Communication protocol used between PX4 and ROS2. It can be installed as stated in [PX4 documentation](https://docs.px4.io/main/en/middleware/uxrce_dds.html#install-standalone-from-source).

### 4. Plotjuggler
For debugging data install [plotjuggler](https://github.com/facontidavide/PlotJuggler) for ROS2  can be installed from snap store using the following commands.
```bash
# installation
sudo snap install plotjuggler

# to launch just run the following command
plotjuggler
```

### 5. Python Packages
```bash
# Install packages
pip install numpy==1.16.0 scipy==1.15.0

# Verify installation
python -c "import numpy, scipy; print(f'numpy: {numpy.__version__}, scipy: {scipy.__version__}')"
```

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone --recursive https://github.com/yash27agarwal/TVC_ws.git

# Alternatively, clone and then initialize
# git clone https://github.com/yash27agarwal/TVC_ws.git
# git submodule update --init --recursive

cd TVC_ws

# Install ROS 2 dependencies
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# Copy TVC model file from src/tvc_controller/models/tvc to PX4_tvc/Tools/simulation/gz/models/
cp -r ./src/tvc_controller/models/tvc ./PX4_tvc/Tools/simulation/gz/models/
```

### 2. Build the PX4 Workspace
First step is to build PX4.
```bash
cd PX4_tvc
make px4_sitl_default

# Try to run it
# This command to successfully open GZ simualator with TVC platform at (0,0,0) coordinates 
PX4_SYS_AUTOSTART=6002  ./build/px4_sitl_default/bin/px4
```

### 3. Connect to PX4
Ensure PX4 is running with the uXRCE-DDS bridge:
```bash
# For SITL
# Open a new terminal
MicroXRCEAgent udp4 -p 8888

# for my machine
micro-xrce-dds-agent udp4 -p 8888
```

### 4. Bind GZ clock to ROS2
```bash
# Bind GZ clock to ros2
# Open a new terminal
ros2 run ros_gz_bridge parameter_bridge /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock

# for ground truth odometry
ros2 run ros_gz_bridge parameter_bridge /model/tvc_0/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry
ros2 run ros_gz_bridge parameter_bridge /model/x500_0/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry

# or run ros_gz_bridge from a yaml file
ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=bridge.yaml
```

### 5. Run the ROS2 Code
```bash
# Open a new terminal
# if you are in PX4_tvc, navigate back to TVC_ws
cd ..

# Build the ROS2 workspace
colcon build

# Source the setup file
source install/setup.bash

# Launch the TVC controller
ros2 launch tvc_controller lqr.launch.py
```

## 🎮 Usage

### TVC Controller Node
The main controller node (`lqr_controller_node.py`) provides:
- **LQR-based control**: Linear Quadratic Regulator for optimal control
- **PX4 integration**: Direct communication with PX4 autopilot
- **Thrust vectoring**: Precise control of thrust magnitude and direction
- **Attitude control**: 6-DOF attitude and position control

### Key Features
- **Inverted coaxial design**: Specialized for inverted coaxial drones
- **Gazebo simulation**: Complete TVC drone model with meshes
- **YAML configuration**: Configurable physical and control parameters
- **Real-time control**: Low-latency communication with PX4

## 📊 Coordinate Frames
This implementation uses standard aerospace coordinate conventions:
- **NED Frame**: North-East-Down for position and linear velocity
- **FRD Frame**: Forward-Right-Down for angular rates and body frame

## 🛠️ TVC Controller Components

### LQR Controller (`lqr.py`)
- **State space model**: 12-state system (position, velocity, orientation, angular rates)
- **Control inputs**: 4 inputs (servo 0, servo 1, total thrust, differential torques)
- **Optimal control**: Minimizes quadratic cost function
- **Physical parameters**: Mass, inertia, geometric properties

### Controller Node (`lqr_controller_node.py`)
- **ROS 2 integration**: Publisher/subscriber architecture
- **PX4 communication**: uXRCE-DDS bridge compatibility
- **Parameter management**: YAML-based configuration
- **Safety features**: Timeout handling and error checking

### Gazebo Model (`models/tvc/`)
- **Complete TVC drone**: SDF model with inertial properties
- **Coaxial propellers**: CW/CCW propeller meshes
- **Sensor integration**: IMU and other sensors
- **Visual representation**: 3D meshes and materials

## 🐛 Debugging
For debugging the data plotjuggler is highly recommended. Use the following command to run plotjuggler.
```bash
plotjuggler
```
In plotjuggler, import the ulog file from PX4 build. In v1.15.4 ulog files are stored in `<PX4>/build/px4_sitl_default/rootfs/log`. Then navigate to your required ulog file and import it to visulaize data. 

## ⚙️ Configuration

The TVC controller is configured through the `src/tvc_controller/config/tvc_params.yaml` file, which contains all the physical, control, and operational parameters for the thrust vector control system.

>**Note:** Make sure to match the physical properties to tvc sdf model.

## 🚁 Modified PX4 (`PX4_tvc/`)

This workspace includes a specialized PX4 fork with modifications for TVC systems:

### Key Modifications
- **Inverted coaxial airframe**: Custom airframe configuration
- **Enhanced ESC interface**: Improved Gazebo simulation
- **Gimbal integration**: Servo control for thrust vectoring
- **Control allocation**: Specialized mixer for coaxial motors

### ⚠️ Important Notes
- **Incompatible with standard PX4**: This modified version is specifically for TVC applications
- **Git tag checks disabled**: For development convenience
- **Custom airframes only**: Only works with inverted coaxial configurations

>**Note:** For more information refer `./PX4_tvc/README.md`

## 📚 Additional Resources

- [PX4 Documentation](https://docs.px4.io/)
- [ROS 2 Documentation](https://docs.ros.org/en/humble/)

---

**Maintainer**: yash.27.agarwal@gmail.com  
**Last Updated**: July 2025

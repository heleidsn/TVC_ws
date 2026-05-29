# PX4 Thrust Vector Control (TVC) Workspace

[![ROS 2](https://img.shields.io/badge/ROS-2-blue)](https://docs.ros.org/en/humble/index.html)
[![PX4](https://img.shields.io/badge/PX4-v1.16-green)](https://px4.io/)
[![Gazebo](https://img.shields.io/badge/Gazebo-Harmonic-orange)](https://gazebosim.org/)

This ROS 2 workspace provides a complete thrust vector control (TVC) implementation for inverted coaxial drones with PX4 autopilot integration. The workspace includes a modified PX4 autopilot, ROS 2 communication bridges, message definitions, and a custom LQR controller for precise thrust vector control of coaxial motor systems with gimbal-based thrust vectoring.

> **Version notice:** This workspace has **officially migrated to PX4 v1.16**. The legacy `PX4_tvc` fork (based on PX4 v1.15.4) is no longer used. Use the `PX4-TVC-NUS` submodule with airframe `6003` and matching `px4_msgs` / `px4_ros_com` branches (`release/1.16`).

![](/assets/test_1.gif)
![](/assets/test_2.gif)

## 🏗️ Workspace Structure

```
TVC_ws/
├── README.md                    # This documentation
├── assets/                      # Media files and documentation assets
│   ├── test_1.gif              # TVC demonstration video 1
│   └── test_2.gif              # TVC demonstration video 2
├── PX4-TVC-NUS/                 # Modified PX4 v1.16 autopilot for TVC (submodule)
│   ├── src/                     # PX4 source code
│   ├── msg/                     # PX4 message definitions
│   ├── Tools/simulation/gz/     # Gazebo Harmonic simulation (includes TVC model)
│   └── ...                      # Standard PX4 structure
└── src/
    ├── px4_msgs/                # PX4 message definitions for ROS 2 (release/1.16)
    ├── px4_ros_com/             # PX4-ROS 2 communication bridge (release/1.16)
    └── tvc_controller/          # Main TVC controller package
        ├── config/
        │   ├── tvc_params.yaml          # Controller parameters
        │   └── px4_rviz.rviz            # RViz2 configuration
        ├── launch/
        │   ├── tvc.launch.py              # Unified launch: controller + bridges + RViz2
        │   ├── lqr.launch.py            # LQR controller only (legacy)
        │   ├── px4_rviz.launch.py       # Visualization only (legacy)
        │   └── gz_vision.launch.py      # Gazebo vision publisher only (legacy)
        ├── models/
        │   └── tvc/                     # Gazebo TVC drone model (reference copy)
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
- **PX4**: **v1.16** via `PX4-TVC-NUS` submodule (SITL or hardware)
- **GZ**: Harmonic (recommended) or Ionic

### Dependencies
- ROS 2 (with `colcon` build tools)
- PX4 Autopilot with uXRCE-DDS bridge
- Python packages:
  - `numpy 1.16.0`
  - `scipy 1.15.0`

### Submodule Branches (v1.16)
| Submodule | Branch |
|---|---|
| `PX4-TVC-NUS` | `px4-tvc-nus-v1.16` |
| `src/px4_msgs` | `release/1.16` |
| `src/px4_ros_com` | `release/1.16` |

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
git clone --recursive https://github.com/heleidsn/TVC_ws.git

# Alternatively, clone and then initialize submodules
# git clone https://github.com/heleidsn/TVC_ws.git
# git submodule update --init --recursive

cd TVC_ws

# Install ROS 2 dependencies
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

> **Note:** The TVC Gazebo model is bundled inside `PX4-TVC-NUS/Tools/simulation/gz/models/tvc/`. You no longer need to manually copy model files from `src/tvc_controller/models/tvc`.

### 2. Build PX4 and ROS 2

Build PX4 first (one-time, or after PX4 code changes):

```bash
cd PX4-TVC-NUS
make px4_sitl
cd ..
```

Build the ROS 2 workspace:

```bash
colcon build
source install/setup.bash
```

### 3. Launch Everything

The unified launch file starts **MicroXRCEAgent**, **PX4 SITL + Gazebo**, the LQR controller, Gazebo bridges, visualization bridge, external vision publisher, and RViz2:

```bash
ros2 launch tvc_controller tvc.launch.py
```

| Launch Argument | Default | Description |
|---|---|---|
| `launch_px4_sitl` | `true` | Start PX4 SITL with Gazebo |
| `launch_microxrce_agent` | `true` | Start MicroXRCE-DDS agent |
| `px4_sys_autostart` | `6003` | TVC airframe id |
| `px4_sim_model` | `tvc` | Gazebo model name |
| `px4_gz_world` | `default` | Gazebo world |
| `microxrce_port` | `8888` | uXRCE-DDS UDP port |
| `ros_startup_delay` | `5.0` | Delay before ROS nodes start (seconds) |

Optional flags:

```bash
# Skip RViz2 (headless)
ros2 launch tvc_controller tvc.launch.py launch_rviz:=false

# PX4 / agent already running externally — ROS stack only
ros2 launch tvc_controller tvc.launch.py \
  launch_px4_sitl:=false launch_microxrce_agent:=false ros_startup_delay:=0

# Controller only (no visualization / Gazebo bridges)
ros2 launch tvc_controller tvc.launch.py \
  launch_rviz:=false launch_gz_bridge:=false \
  launch_px4_rviz_bridge:=false launch_gz_vision:=false
```

<details>
<summary>Manual step-by-step (legacy)</summary>

If you prefer to run components separately:

```bash
# Terminal 1: PX4 SITL
cd PX4-TVC-NUS
PX4_SYS_AUTOSTART=6003 PX4_SIM_MODEL=tvc PX4_GZ_WORLD=default ./build/px4_sitl_default/bin/px4

# Terminal 2: MicroXRCE-DDS agent
MicroXRCEAgent udp4 -p 8888

# Terminal 3: ROS stack without PX4 / agent
ros2 launch tvc_controller tvc.launch.py \
  launch_px4_sitl:=false launch_microxrce_agent:=false ros_startup_delay:=0
```

Or run individual components:

```bash
# Gazebo bridges
ros2 run ros_gz_bridge parameter_bridge --ros-args \
  -p config_file:=$(ros2 pkg prefix tvc_controller)/share/tvc_controller/config/bridge.yaml

# LQR controller
ros2 launch tvc_controller lqr.launch.py

# Visualization
ros2 launch tvc_controller px4_rviz.launch.py
```

</details>

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
In plotjuggler, import the ulog file from PX4 build. For PX4 v1.16, ulog files are stored in `PX4-TVC-NUS/build/px4_sitl_default/rootfs/log`. Then navigate to your required ulog file and import it to visulaize data. 

## ⚙️ Configuration

The TVC controller is configured through the `src/tvc_controller/config/tvc_params.yaml` file, which contains all the physical, control, and operational parameters for the thrust vector control system.

>**Note:** Make sure to match the physical properties to tvc sdf model.

## 🚁 Modified PX4 (`PX4-TVC-NUS/`)

This workspace includes a PX4 v1.16 fork with modifications for TVC systems:

### Key Modifications
- **TVC airframe `6003_tvc`**: Custom airframe configuration for Gazebo Harmonic
- **Bundled TVC Gazebo model**: Located at `Tools/simulation/gz/models/tvc/`
- **Gimbal integration**: Servo control for thrust vectoring
- **Control allocation**: Specialized mixer for coaxial motors

### ⚠️ Important Notes
- **Incompatible with standard PX4**: This modified version is specifically for TVC applications
- **Requires v1.16 message bridge**: Use `px4_msgs` and `px4_ros_com` on `release/1.16`
- **Custom airframes only**: Only works with inverted coaxial TVC configurations

>**Note:** For more information refer to `./PX4-TVC-NUS/README.md`

## 🔄 Migration from v1.15 (`PX4_tvc`)

If you are upgrading from the older setup:

| Item | v1.15 (legacy) | v1.16 (current) |
|---|---|---|
| PX4 fork | `PX4_tvc` | `PX4-TVC-NUS` |
| PX4 version | v1.15.4 | v1.16 |
| Airframe ID | `6002` | `6003` |
| `px4_msgs` / `px4_ros_com` | `release/1.15` | `release/1.16` |
| TVC model setup | Manual copy to PX4 tree | Bundled in `PX4-TVC-NUS` |

After pulling the latest changes, re-initialize submodules:

```bash
git submodule update --init --recursive
```

## 📚 Additional Resources

- [PX4 Documentation](https://docs.px4.io/)
- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [PX4-TVC-NUS Repository](https://github.com/heleidsn/PX4-TVC-NUS)

---

**Maintainer**: yash.27.agarwal@gmail.com  
**Last Updated**: May 2026

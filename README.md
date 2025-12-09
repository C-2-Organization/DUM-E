# DRIP COFFEE MAKER

(Introduction)

---

## How to get started

### Setup environment

**Recommended OS**

- Ubuntu 22.04 (ROS2 Humble)
- Python ^3.10
- Node.js ^18.x

**Essential packages**

```bash
# common
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip

# ROS2 (Humble)
# https://docs.ros.org/en/humble/Installation.html

# ROS build tool
sudo apt install -y build-essential cmake python3-colcon-common-extensions python3-vcstool
```

### Clon repo

```bash
git clone https://github.com/C-2-Organization/DUM-E.git
cd DUM-E
```

### ROS2

1. Install dependency

    ```bash
    cd ros2_ws
    mkdir -p src
    vcs import src < ros2_ws.repos
    ```

2. Build

    ```bash
    cd ros2_ws
    colcon build
    source install/setup.bash
    ```

3. Connenct to robot

    ```bash
    # virtual mode
    ros2 launch  dsr_bringup2 dsr_bringup2_rviz.launch.py mode:=virtual host:=127.0.0.1 port:=12345 model:=m0609

    # real mode
    ros2 launch  dsr_bringup2 dsr_bringup2_rviz.launch.py mode:=real  host:=192.168.1.100 port:=12345 model:=m0609
    ```

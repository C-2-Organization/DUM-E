# DUM-E

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
cd ~/DUM-E
```

### Install dependencies

```bash
cd ~/DUM-E
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

```bash
cd ~/DUM-E/ros2_ws
vcs import src < ros2_ws.repos
```

### Build & Run ROS2 Packages

1. Build

    ```bash
    cd ros2_ws
    colcon build
    source install/setup.bash
    ```

2. Run Jarvis & Dummy

    You can run Jarvis & Dummy by running the following commands.

    ```bash
    # Jarvis
    cd ~/DUM-E/services/audio_io
    uvicorn app.main:app --reload

    # Dummy
    ros2 launch dum_e_bringup dum_e_bringup.launch.py
    ```

    *You can also run Dummy with your voice command (such as "Wake up Dummy" or "Wake up robot") via Jarvis.*


3. Connenct to robot manually

    If you want to connect to robot manually, run one of the following commands.

        ```bash
        # virtual mode
        ros2 launch  dsr_bringup2 dsr_bringup2_rviz.launch.py mode:=virtual host:=127.0.0.1 port:=12345 model:=m0609

        # real mode
        ros2 launch  dsr_bringup2 dsr_bringup2_rviz.launch.py mode:=real  host:=192.168.1.100 port:=12345 model:=m0609
        ```

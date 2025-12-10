from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from pathlib import Path
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # -------------------------------
    # Launch arguments (Doosan 옵션)
    # -------------------------------
    doosan_host = LaunchConfiguration("doosan_host")
    doosan_port = LaunchConfiguration("doosan_port")
    doosan_mode = LaunchConfiguration("doosan_mode")
    doosan_model = LaunchConfiguration("doosan_model")

    declare_doosan_host = DeclareLaunchArgument(
        "doosan_host",
        default_value="192.168.1.100",
        description="Doosan controller IP",
    )
    declare_doosan_port = DeclareLaunchArgument(
        "doosan_port",
        default_value="12345",
        description="Doosan controller port",
    )
    declare_doosan_mode = DeclareLaunchArgument(
        "doosan_mode",
        default_value="real",
        description="Doosan mode (real/simulation)",
    )
    declare_doosan_model = DeclareLaunchArgument(
        "doosan_model",
        default_value="m0609",
        description="Doosan robot model",
    )

    # -------------------------------
    # 1) Doosan bringup
    #    ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py ...
    # -------------------------------
    dsr_bringup_pkg_share = Path(get_package_share_directory("dsr_bringup2"))

    dsr_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(dsr_bringup_pkg_share / "launch" / "dsr_bringup2_rviz.launch.py")
        ),
        launch_arguments={
            "mode": doosan_mode,
            "host": doosan_host,
            "port": doosan_port,
            "model": doosan_model,
        }.items(),
    )

    # -------------------------------
    # 2) RealSense bringup
    #    보통 rs_align_depth_launch.py 대신 rs_launch.py 가 있음
    #    ros2 launch realsense2_camera rs_launch.py depth_module.depth_profile:=640x480x30 ...
    # -------------------------------
    rs_pkg_share = Path(get_package_share_directory("realsense2_camera"))

    rs_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(rs_pkg_share / "launch" / "rs_launch.py")
        ),
        launch_arguments={
            "depth_module.depth_profile": "640x480x30",
            "rgb_camera.color_profile": "640x480x30",
            "initial_reset": "true",
            "align_depth.enable": "true",
            "pointcloud.enable": "true",
        }.items(),
    )

    # -------------------------------
    # 3) Perception node
    #    ros2 run dum_e_perception perception_node
    # -------------------------------
    perception_node = Node(
        package="dum_e_perception",
        executable="perception_node",
        name="dum_e_perception",
        output="screen",
    )

    # -------------------------------
    # 4) Skill manager node
    #    ros2 run dum_e_motion skill_manager_node
    # -------------------------------
    skill_manager_node = Node(
        package="dum_e_motion",
        executable="skill_manager_node",
        name="skill_manager_node",
        output="screen",
    )

    # -------------------------------
    # LaunchDescription 구성
    # -------------------------------
    ld = LaunchDescription()

    # Doosan 인자 선언
    ld.add_action(declare_doosan_host)
    ld.add_action(declare_doosan_port)
    ld.add_action(declare_doosan_mode)
    ld.add_action(declare_doosan_model)

    # bringup 요소들
    ld.add_action(dsr_launch)
    ld.add_action(rs_launch)
    ld.add_action(perception_node)
    ld.add_action(skill_manager_node)

    return ld

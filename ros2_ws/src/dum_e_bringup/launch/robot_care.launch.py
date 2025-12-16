from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 튜닝 파라미터 (기본값 빠르게 설정)
    monitor_interval_arg = DeclareLaunchArgument(
        'monitor_interval', default_value='0.2',
        description='Robot state polling interval (sec)'
    )
    failure_threshold_arg = DeclareLaunchArgument(
        'failure_threshold', default_value='8',
        description='Consecutive failures before restart'
    )
    initial_grace_arg = DeclareLaunchArgument(
        'initial_safe_stop_grace', default_value='1.5',
        description='Grace seconds before initial SAFE_STOP recovery'
    )

    # 독립 실행 복구 노드 (권장)
    care = Node(
        package='dum_e_bringup',
        executable='robot_state_care_node',
        name='robot_state_care_node',
        output='screen',
        parameters=[{
            'monitor_interval': LaunchConfiguration('monitor_interval'),
            'failure_threshold': LaunchConfiguration('failure_threshold'),
            'initial_safe_stop_grace': LaunchConfiguration('initial_safe_stop_grace'),
        }]
    )

    ld = LaunchDescription([
        monitor_interval_arg,
        failure_threshold_arg,
        initial_grace_arg,
        care,
    ])
    return ld

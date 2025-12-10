#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class FetchObjectNode(Node):
    """
    Node that fetches a specific object.
    Currently hardcoded to fetch 'scissors'.
    """

    def __init__(self):
        super().__init__('fetch_object_node')
        
        # Hardcoded object to fetch
        self.target_object = 'scissors'
        
        # Publisher for object fetch commands
        self.publisher_ = self.create_publisher(
            String,
            'fetch_object_command',
            10
        )
        
        # Timer to periodically send fetch command
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info(f'Fetch Object Node started. Target object: {self.target_object}')

    def timer_callback(self):
        """Send fetch command periodically"""
        msg = String()
        msg.data = f'fetch:{self.target_object}'
        self.publisher_.publish(msg)
        self.get_logger().debug(f'Published fetch command for: {self.target_object}')

    def fetch_object(self, object_name: str) -> bool:
        """
        Fetch a specific object.
        
        Args:
            object_name: Name of the object to fetch
            
        Returns:
            bool: True if fetch successful, False otherwise
        """
        self.get_logger().info(f'Fetching object: {object_name}')
        
        # TODO: Implement actual object fetching logic
        # This would involve:
        # 1. Object detection/localization
        # 2. Motion planning
        # 3. Gripper control
        # 4. Navigation if needed
        
        return True


def main(args=None):
    rclpy.init(args=args)
    node = FetchObjectNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Fetch Object Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

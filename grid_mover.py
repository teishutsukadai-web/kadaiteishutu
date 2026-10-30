import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import tf_transformations

class Pure2DSpaceMover(Node):
    def __init__(self):
        super().__init__('pure_2d_space_mover')
        
        # パブリッシャーとサブスクライバー
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        
        # タイマー（0.1秒ごとに制御ループ）
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # 現在の位置と向き
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        # センサー情報
        self.front_min_dist = 999.0
        self.left_clearance = 0.0
        self.right_clearance = 0.0
        
        # 空間内を広く巡回するウェイポイント
        self.waypoints = [
            (1.5, 0.0),
            (1.5, 1.5),
            (0.0, 1.5),
            (0.0, 0.0)
        ]
        self.current_waypoint_index = 0
        self.goal_reached_threshold = 0.15

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        orientation_q = msg.pose.pose.orientation
        _, _, self.current_yaw = tf_transformations.euler_from_quaternion([
            orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w
        ])

    def scan_callback(self, msg):
        num_readings = len(msg.ranges)
        if num_readings == 0:
            return

        # 正面（±20度）の最小距離
        front_ranges = msg.ranges[:20] + msg.ranges[-20:]
        valid_front = [r for r in front_ranges if not math.isnan(r) and r > msg.range_min and r < msg.range_max]
        self.front_min_dist = min(valid_front) if valid_front else 999.0

        # 左右の空きスペース（平均距離）
        left_ranges = msg.ranges[30:90]
        right_ranges = msg.ranges[270:330]
        valid_left = [r for r in left_ranges if not math.isnan(r)]
        valid_right = [r for r in right_ranges if not math.isnan(r)]
        
        self.left_clearance = sum(valid_left) / len(valid_left) if valid_left else 0.0
        self.right_clearance = sum(valid_right) / len(valid_right) if valid_right else 0.0

    def control_loop(self):
        msg = Twist()

        # 【重要】縦回転（ピッチ・ロール）や横方向への意図しないずらしを完全にゼロに固定
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0

        # 【ギリギリ回避モード（横回転のみでスマートに回避）】
        if self.front_min_dist < 0.22:
            self.get_logger().warn(f'障害物接近！ ギリギリ回避 (距離: {self.front_min_dist:.2f}m)', throttle_duration_sec=0.5)
            msg.linear.x = 0.05  # 微小な前進を維持
            
            # 広い方へ「横回転（ヨー軸の旋回）」のみで一気に方向転換
            if self.left_clearance >= self.right_clearance:
                msg.angular.z = 1.4  # 左への横回転
            else:
                msg.angular.z = -1.4 # 右への横回転
                
            self.publisher_.publish(msg)
            return

        # すべての巡回ポイントを終了した場合
        if self.current_waypoint_index >= len(self.waypoints):
            self.get_logger().info('空間の巡回を完了しました！', throttle_duration_sec=5.0)
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.publisher_.publish(msg)
            return

        # 【通常空間移動モード】
        target_x, target_y = self.waypoints[self.current_waypoint_index]
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        distance = math.sqrt(dx**2 + dy**2)
        target_yaw = math.atan2(dy, dx)

        angle_error = target_yaw - self.current_yaw
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

        if distance < self.goal_reached_threshold:
            self.get_logger().info(f'ポイント {self.current_waypoint_index + 1} 到着')
            self.current_waypoint_index += 1
            return

        # 向きが大きくずれているときはその場で横回転、近ければ前進しながら横回転
        if abs(angle_error) > 0.3:
            msg.linear.x = 0.0
            msg.angular.z = 1.0 * angle_error  # その場で横回転
        else:
            msg.linear.x = 0.22              # 前進
            msg.angular.z = 1.5 * angle_error  # 横回転の調整

        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = Pure2DSpaceMover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
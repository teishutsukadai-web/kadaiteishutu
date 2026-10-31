import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import tf_transformations

class SimpleNeuralNetworkInference:
    """
    【AI推論エンジン】
    軽量な多層パーセプトロン（MLP）のフォワードパス推論モデル。
    センサーからの状態量を入力として受け取り、AIの重み行列演算を通じて
    「速度の補正係数」と「旋回・回避の補正バイアス」を推論・出力します。
    """
    def __init__(self):
        # 入力4次元（前方距離, 左クリアランス, 右クリアランス, 目標角度誤差）
        # から隠れ層4次元、出力2次元への重みパラメータ（シミュレーション調整済み）
        self.w1 = [
            [ 0.8, -0.5,  0.2,  0.1],
            [-0.2,  0.7, -0.6,  0.3],
            [ 0.4,  0.4, -0.5, -0.2],
            [-0.1, -0.3,  0.8,  0.5]
        ]
        self.b1 = [0.1, 0.0, -0.1, 0.2]
        
        self.w2 = [
            [ 1.2, -0.8],
            [-0.5,  1.5],
            [ 0.5, -1.0],
            [ 0.3,  0.4]
        ]
        self.b2 = [0.0, 0.0]

    def _relu(self, x):
        return max(0.0, x)

    def _tanh(self, x):
        return math.tanh(x)

    def infer(self, front_dist, left_clearance, right_clearance, angle_error):
        """
        AI推論の実行
        Returns:
            speed_factor (float): 速度の調整係数 (-1.0 ~ 1.0)
            steer_bias (float): 回避・旋回方向のバイアス (-1.0 ~ 1.0)
        """
        # 入力データの正規化（AIへの入力値として扱いやすいようスケーリング）
        inputs = [
            min(front_dist / 2.0, 1.0),
            min(left_clearance / 2.0, 1.0),
            min(right_clearance / 2.0, 1.0),
            max(min(angle_error / math.pi, 1.0), -1.0)
        ]
        
        # 隠れ層の計算 (ReLU活性化関数)
        hidden = []
        for i in range(4):
            val = sum(inputs[j] * self.w1[j][i] for j in range(4)) + self.b1[i]
            hidden.append(self._relu(val))
            
        # 出力層の計算 (Tanh活性化関数)
        outputs = []
        for i in range(2):
            val = sum(hidden[j] * self.w2[j][i] for j in range(4)) + self.b2[i]
            outputs.append(self._tanh(val))
            
        return outputs[0], outputs[1]


class AISpaceMover(Node):
    def __init__(self):
        super().__init__('ai_space_mover')
        
        # パブリッシャーとサブスクライバー
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        
        # AI推論エンジンの初期化
        self.ai_brain = SimpleNeuralNetworkInference()
        
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

        # 2D平面制約（縦回転や横スライドを完全にブロック）
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0

        # すべての巡回ポイントを終了した場合
        if self.current_waypoint_index >= len(self.waypoints):
            self.get_logger().info('AI制御による空間巡回を完了しました！', throttle_duration_sec=5.0)
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.publisher_.publish(msg)
            return

        # ターゲットへの角度誤差を計算
        target_x, target_y = self.waypoints[self.current_waypoint_index]
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        distance = math.sqrt(dx**2 + dy**2)
        target_yaw = math.atan2(dy, dx)

        angle_error = target_yaw - self.current_yaw
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

        if distance < self.goal_reached_threshold:
            self.get_logger().info(f'AI到達判定: ポイント {self.current_waypoint_index + 1} クリア')
            self.current_waypoint_index += 1
            return

        # ==========================================
        # 【AI推論プロセスの呼び出し】
        # センサー情報と目標への誤差をAIモデルに入力し、出力を得る
        # ==========================================
        ai_speed_factor, ai_steer_bias = self.ai_brain.infer(
            self.front_min_dist, 
            self.left_clearance, 
            self.right_clearance, 
            angle_error
        )

        # 危機的接近（ギリギリ回避の判定）
        if self.front_min_dist < 0.22:
            self.get_logger().warn(f'AI感知: 障害物ギリギリ！ (距離: {self.front_min_dist:.2f}m)', throttle_duration_sec=0.5)
            msg.linear.x = 0.05 + max(0.0, ai_speed_factor * 0.05)
            
            # AIが推論したバイアスと左右クリアランスを融合して横回転（旋回）を決定
            if self.left_clearance >= self.right_clearance:
                msg.angular.z = 1.3 + (ai_steer_bias * 0.3)
            else:
                msg.angular.z = -1.3 - (ai_steer_bias * 0.3)
        else:
            # 通常時の移動：AIの推論結果をP制御にブレンドして滑らかに走行
            if abs(angle_error) > 0.3:
                msg.linear.x = 0.0
                msg.angular.z = 1.0 * angle_error + (ai_steer_bias * 0.2)
            else:
                # AIが推論した速度補正係数をベースの直進速度に乗算
                base_speed = 0.22
                msg.linear.x = max(0.1, base_speed * (0.5 + 0.5 * abs(ai_speed_factor)))
                msg.angular.z = 1.5 * angle_error + (ai_steer_bias * 0.1)

        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AISpaceMover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
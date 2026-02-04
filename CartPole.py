import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

class InvertedPendulumEnv:
    """修正后的倒立摆仿真环境（正确的角度定义）"""
    
    def __init__(self, dt=0.02, max_force=100.0):
        """
        初始化倒立摆环境
        
        参数:
        dt: 仿真时间步长 (秒)
        max_force: 最大控制力 (N)
        """
        # 物理参数
        self.m_cart = 1.0     # 小车质量 (kg)
        self.m_pole = 0.3     # 摆杆质量 (kg) - 稍微增加以增加惯性
        self.l = 0.5          # 摆杆长度 (m) - 质心到铰链的距离
        self.g = 9.81         # 重力加速度 (m/s^2)
        self.dt = dt          # 时间步长
        self.max_force = max_force  # 最大控制力
        
        # 状态维度
        self.state_dim = 4    # [位置, 速度, 角度, 角速度]
        self.control_dim = 1  # [水平力]
        
        # 状态限制
        self.pos_limit = 2.0      # 位置限制 (m)
        self.vel_limit = 5.0      # 速度限制 (m/s)
        self.angle_limit = np.pi  # 角度限制 (rad)
        self.omega_limit = 10.0   # 角速度限制 (rad/s)
        
        # 当前状态
        self.state = np.zeros((self.state_dim,1))
        
        # 仿真时间
        self.time = 0.0
        
        # 用于记录轨迹
        self.state_history = []
        self.control_history = []
        self.time_history = []
        
        # 系统参数计算（用于动力学方程）
        self.M = self.m_cart + self.m_pole  # 总质量
        self.I = (1/3) * self.m_pole * self.l**2  # 摆杆绕端点的转动惯量
        
    def reset(self, initial_state=None):
        """
        重置环境到初始状态
        
        参数:
        initial_state: 初始状态，如果为None则使用随机初始状态
        """
        if initial_state is None:
            # 随机初始状态：小车在中心附近，摆杆稍微偏离垂直向上
            # θ=0 是垂直向上，所以我们从接近垂直向上开始
            self.state = np.array([
                np.random.uniform(-0.1, 0.1),  # 位置
                np.random.uniform(-0.1, 0.1),  # 速度
                np.random.uniform(-0.1, 0.1),  # 角度 (接近0，即接近垂直向上)
                np.random.uniform(-0.1, 0.1)   # 角速度
            ])
        else:
            self.state = np.array(initial_state)
            
        # 确保角度在[-π, π]范围内
        self.state[2] = self.normalize_angle(self.state[2])
            
        # 重置记录
        self.time = 0.0
        self.state_history = [self.state.copy()]
        self.control_history = []
        self.time_history = [self.time]
        
        return self.state.copy()
    
    def dynamics(self, t, state, u):
        """
        连续时间动力学方程（正确的角度定义）

        参数:
        t: 时间 (未使用，但solve_ivp需要)
        state: 当前状态 [p, v, θ, ω]
              θ=0: 摆杆垂直向上
              θ>0: 顺时针旋转（向右倾斜）
              θ<0: 逆时针旋转（向左倾斜）
        u: 控制输入 [F]

        返回:
        state_dot: 状态导数
        """
        p, v, theta, omega = state
        F = np.clip(u[0], -self.max_force, self.max_force)  # 确保控制输入在限制范围内

        # 提取参数
        m_c = self.m_cart
        m_p = self.m_pole
        l = self.l
        g = self.g

        # 三角函数
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        denom = m_c + m_p * sin_theta**2

        x_acc = (F + m_p * sin_theta * (l * omega**2 + g * cos_theta)) / denom
        theta_acc = (-F * cos_theta - (m_c + m_p) * g * sin_theta - m_p * l * omega**2 * sin_theta * cos_theta) / (l * denom)


        return np.array([v, x_acc, omega, theta_acc])
    
    
    def normalize_angle(self, angle):
        """将角度归一化到 [-π, π]"""
        angle = angle % (2 * np.pi)
        if angle > np.pi:
            angle -= 2 * np.pi
        return angle
    
    def step(self, u, integration_method='rk4'):
        """
        执行一步仿真
        
        参数:
        u: 控制输入
        integration_method: 积分方法 ('euler' 或 'rk4')
        
        返回:
        next_state: 下一个状态
        done: 是否结束
        info: 附加信息
        """
        # 限制控制输入
        u_clipped = np.clip(u, -self.max_force, self.max_force)
        
        # 记录控制输入
        self.control_history.append(u_clipped.copy())
        
        # 根据选择的积分方法更新状态
        if integration_method == 'euler':
            state_dot = self.dynamics(self.time, self.state, u_clipped)
            next_state = self.state + state_dot * self.dt
        elif integration_method == 'rk4':
            k1 = self.dynamics(self.time, self.state, u_clipped)
            k2 = self.dynamics(self.time + self.dt/2, self.state + k1*self.dt/2, u_clipped)
            k3 = self.dynamics(self.time + self.dt/2, self.state + k2*self.dt/2, u_clipped)
            k4 = self.dynamics(self.time + self.dt, self.state + k3*self.dt, u_clipped)
            next_state = self.state + (k1 + 2*k2 + 2*k3 + k4) * self.dt / 6
        else:
            raise ValueError("integration_method must be 'euler' or 'rk4'")
        
        # 限制状态（防止数值不稳定）
        next_state[0] = np.clip(next_state[0], -self.pos_limit, self.pos_limit)
        next_state[1] = np.clip(next_state[1], -self.vel_limit, self.vel_limit)
        next_state[3] = np.clip(next_state[3], -self.omega_limit, self.omega_limit)
        
        # 角度归一化到 [-π, π]
        next_state[2] = self.normalize_angle(next_state[2])
        
        # 更新状态和时间
        self.state = next_state
        self.time += self.dt
        
        # 记录
        self.state_history.append(self.state.copy())
        self.time_history.append(self.time)
        
        # 检查是否结束（超出位置限制或角度过大）
        # 我们允许角度在[-π, π]之间，但直立平衡时θ接近0
        done = False

        angle_from_upright = abs(self.normalize_angle(self.state[2]))
        done = (np.abs(self.state[0]) > self.pos_limit * 0.9) or \
            (angle_from_upright > np.pi/2)  # 偏离垂直方向超过90度
    
        info = {}
        
        return self.state.copy(), done, info
    
    def GetDerivatives(self,state,u):
        p = state[0]
        v = state[1]
        M = self.m_cart
        m = self.m_pole
        l = self.l
        g = 9.8
        theta = state[2]
        omega = state[3]
        A = np.zeros((4, 4))
        B = np.zeros((4, 1))
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        u = 0
        A[0,1] = 1
        A[2,3] = 1
        den = M + m * sin_theta * sin_theta
        term_1_2_num_part1 = 2*m*(m*(g * cos_theta + l * omega * omega)*sin_theta + u)*sin_theta*cos_theta
        term_1_2_num_part2 = -g * m * sin_theta *sin_theta + m * (g * cos_theta + l * omega * omega)*cos_theta
        A[1,2] = -term_1_2_num_part1 / np.power(den,2) + term_1_2_num_part2/den
        A[1,3] = 2*l*m*omega*sin_theta / den
        term_3_2_num_part1 = 2*m*(-g*(M+m)*sin_theta-l*m*omega*omega*sin_theta*cos_theta - u*cos_theta)*cos_theta*sin_theta
        term_3_2_num_part2 = -g*(M+m)*cos_theta + l*m*omega*omega*sin_theta*sin_theta - l*m*omega*omega*cos_theta*cos_theta + u * sin_theta
        
        A[3,2] = -term_3_2_num_part1/(l*den*den) + term_3_2_num_part2/(l * den)
        A[3,3] = -2 * m * omega * sin_theta * cos_theta / den
        
        B[1,0] = 1 / den
        B[3,0] = -cos_theta / (l * den)
        return A, B

    def render(self, mode='human', save_path=None):
        """
        渲染当前状态
        
        参数:
        mode: 渲染模式 ('human' 或 'static')
        save_path: 如果提供，保存动画到文件
        """
        if mode == 'static':
            self._render_static()
        elif mode == 'human':
            self._render_animation(save_path)
    
    def _render_static(self):
        """静态渲染（单帧）"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        times, states, controls = self.get_trajectory()
        
        if len(times) == 0:
            return
        
        # 位置轨迹
        axes[0, 0].plot(times, states[:, 0], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Position (m)')
        axes[0, 0].set_title('Cart Position')
        axes[0, 0].grid(True)
        axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 角度轨迹（转换为度数）
        angle_deg = np.degrees(states[:, 2])
        axes[0, 1].plot(times, angle_deg, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Angle (deg)')
        axes[0, 1].set_title('Pendulum Angle (0°=upright)')
        axes[0, 1].grid(True)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 相位图：角度 vs 角速度
        axes[0, 2].plot(states[:, 2], states[:, 3], 'g-', linewidth=1, alpha=0.7)
        axes[0, 2].plot(states[0, 2], states[0, 3], 'go', markersize=10, label='Start')
        axes[0, 2].plot(states[-1, 2], states[-1, 3], 'rx', markersize=10, label='End')
        axes[0, 2].set_xlabel('Angle (rad)')
        axes[0, 2].set_ylabel('Angular Velocity (rad/s)')
        axes[0, 2].set_title('Phase Plot')
        axes[0, 2].grid(True)
        axes[0, 2].legend()
        axes[0, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 2].axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # 速度轨迹
        axes[1, 0].plot(times, states[:, 1], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Velocity (m/s)')
        axes[1, 0].set_title('Cart Velocity')
        axes[1, 0].grid(True)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 控制输入轨迹
        if len(controls) > 0:
            control_times = times[:-1]  # 控制输入在时间步之间
            axes[1, 1].plot(control_times, controls[:, 0], 'm-', linewidth=2)
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Force (N)')
            axes[1, 1].set_title('Control Input')
            axes[1, 1].grid(True)
            axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 系统示意图
        ax_schematic = axes[1, 2]
        ax_schematic.clear()
        ax_schematic.set_xlim(-2, 2)
        ax_schematic.set_ylim(-1, 1)
        ax_schematic.set_aspect('equal')
        ax_schematic.set_title('System Schematic')
        ax_schematic.axis('off')
        
        # 绘制轨道
        ax_schematic.plot([-self.pos_limit, self.pos_limit], [0, 0], 'k-', linewidth=3)
        
        # 绘制小车
        cart_width = 0.3
        cart_height = 0.15
        current_pos = states[-1, 0]
        cart = patches.Rectangle((current_pos - cart_width/2, -cart_height/2), 
                                 cart_width, cart_height, 
                                 fill=True, color='blue', alpha=0.7)
        ax_schematic.add_patch(cart)
        
        # 绘制摆杆
        current_angle = states[-1, 2]
        pole_length = self.l * 2  # 为了可视化更明显
        pole_x = current_pos
        pole_y = 0
        bob_x = pole_x + pole_length * np.sin(current_angle)
        bob_y = pole_y - pole_length * np.cos(current_angle)  # 注意：y轴向上为正
        
        ax_schematic.plot([pole_x, bob_x], [pole_y, bob_y], 'r-', linewidth=3)
        ax_schematic.plot(bob_x, bob_y, 'ro', markersize=10)
        
        # 添加角度标注
        angle_text = f"θ = {current_angle:.2f} rad\n({np.degrees(current_angle):.1f}°)"
        ax_schematic.text(1.0, 0.8, angle_text, fontsize=10, 
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def _render_animation(self, save_path=None):
        """动画渲染"""
        fig = plt.figure(figsize=(12, 6))
        
        # 创建两个子图：动画和状态轨迹
        ax1 = plt.subplot(2, 3, (1, 2))
        ax2 = plt.subplot(2, 3, 4)
        ax3 = plt.subplot(2, 3, 5)
        ax4 = plt.subplot(2, 3, 6)
        
        # 获取轨迹数据
        times, states, controls = self.get_trajectory()
        
        if len(times) == 0:
            return
        
        # 设置动画坐标轴
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-1.2, 1.2)
        ax1.set_aspect('equal')
        ax1.set_xlabel('Position (m)')
        ax1.set_ylabel('Height (m)')
        ax1.set_title('Inverted Pendulum Simulation')
        ax1.grid(True, alpha=0.3)
        
        # 绘制轨道
        track_length = 4.0
        ax1.plot([-track_length/2, track_length/2], [0, 0], 'k-', linewidth=3)
        
        # 初始化小车和摆杆
        cart_width = 0.3
        cart_height = 0.15
        cart = patches.Rectangle((0, 0), cart_width, cart_height, 
                                 fill=True, color='blue', alpha=0.7)
        ax1.add_patch(cart)
        
        pole_length = self.l * 2  # 为了可视化更明显，稍微放大
        pole, = ax1.plot([], [], 'r-', linewidth=3)
        bob, = ax1.plot([], [], 'ro', markersize=10)
        
        # 添加角度标注
        angle_text = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, fontsize=10,
                              verticalalignment='top', 
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 添加垂直线（参考线）
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 设置轨迹图
        ax2.set_xlim(0, times[-1])
        ax2.set_ylim(-track_length/2, track_length/2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position (m)')
        ax2.set_title('Position Trajectory')
        ax2.grid(True, alpha=0.3)
        pos_line, = ax2.plot([], [], 'b-', linewidth=2)
        pos_line_current, = ax2.plot([], [], 'bo', markersize=6)
        
        # 设置角度图
        ax3.set_xlim(0, times[-1])
        ax3.set_ylim(-np.pi/2, np.pi/2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Angle (deg)')
        ax3.set_title('Angle Trajectory (0°=upright)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        angle_line, = ax3.plot([], [], 'r-', linewidth=2)
        angle_line_current, = ax3.plot([], [], 'ro', markersize=6)
        
        # 设置控制图
        ax4.set_xlim(0, times[-1])
        ax4.set_ylim(-self.max_force, self.max_force)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Force (N)')
        ax4.set_title('Control Input')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        control_line, = ax4.plot([], [], 'm-', linewidth=2)
        
        def init():
            """初始化动画"""
            cart.set_xy((-cart_width/2, -cart_height/2))
            pole.set_data([], [])
            bob.set_data([], [])
            angle_text.set_text('')
            pos_line.set_data([], [])
            pos_line_current.set_data([], [])
            angle_line.set_data([], [])
            angle_line_current.set_data([], [])
            control_line.set_data([], [])
            return cart, pole, bob, angle_text, pos_line, pos_line_current, angle_line, angle_line_current, control_line
        
        def animate(i):
            """动画帧更新"""
            # 更新小车位置
            pos = states[i, 0]
            cart.set_xy((pos - cart_width/2, -cart_height/2))
            
            # 更新摆杆位置
            angle = states[i, 2]
            pole_x = pos
            pole_y = 0
            bob_x = pole_x + pole_length * np.sin(angle)
            bob_y = pole_y + pole_length * np.cos(angle)  # y轴向上为正，cos前加负号
            
            pole.set_data([pole_x, bob_x], [pole_y, bob_y])
            bob.set_data([bob_x], [bob_y])
            
            # 更新角度文本
            angle_deg = np.degrees(angle)
            angle_text.set_text(f'θ = {angle:.2f} rad\n({angle_deg:.1f}°)')
            
            # 更新轨迹图
            time_window = times[:i+1]
            pos_window = states[:i+1, 0]
            angle_window_deg = np.degrees(states[:i+1, 2])
            
            pos_line.set_data(time_window, pos_window)
            pos_line_current.set_data([times[i]], [states[i, 0]])
            
            angle_line.set_data(time_window, angle_window_deg/57.3)
            angle_line_current.set_data([times[i]], [angle_window_deg[i]/57.3])
            
            # 更新控制图
            if len(controls) > 0 and i < len(controls):
                control_times = times[:i+1]
                control_window = controls[:i+1, 0]
                control_line.set_data(control_times, control_window)
            
            return cart, pole, bob, angle_text, pos_line, pos_line_current, angle_line, angle_line_current, control_line
        
        # 创建动画
        anim = FuncAnimation(fig, animate, frames=len(times),
                            init_func=init, blit=False, interval=50)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=20)
        
        plt.tight_layout()
        plt.show()
    
    def get_trajectory(self):
        """获取历史轨迹"""
        states = np.array(self.state_history) if self.state_history else np.array([])
        controls = np.array(self.control_history) if self.control_history else np.array([])
        times = np.array(self.time_history) if self.time_history else np.array([])
        return times, states, controls
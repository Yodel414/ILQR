from CartPole import InvertedPendulumEnv
import numpy as np
from ilqr import ILQR
class PDController:
    """简单的PD控制器，用于测试环境"""
    
    def __init__(self, kp_pos=10.0, kd_pos=5.0, kp_angle=100.0, kd_angle=20.0):
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.kp_angle = kp_angle
        self.kd_angle = kd_angle
    
    def compute_control(self, state, target_state=None):
        """
        计算PD控制输入
        
        参数:
        state: 当前状态 [p, v, θ, ω]
        target_state: 目标状态，默认是直立平衡
        
        返回:
        u: 控制输入
        """
        if target_state is None:
            target_state = np.array([0, 0, np.pi, 0])  # 直立平衡
        
        # 计算误差
        error = state - target_state
        
        # 注意：角度的处理，我们想要在π处平衡
        # 将角度误差转换到[-π, π]
        angle_error = ((error[2] + np.pi) % (2 * np.pi)) - np.pi
        
        # PD控制
        u_pos = -self.kp_pos * error[0] - self.kd_pos * error[1]
        u_angle = -self.kp_angle * angle_error - self.kd_angle * error[3]
        
        # 组合控制输入
        u = u_pos + u_angle
        
        return np.array([u])
# 测试和演示

def test_pd_controller():
    """测试PD控制器"""
    print("\n测试PD控制器...")
    
    # 创建环境和控制器
    env = InvertedPendulumEnv(dt=0.01, max_force=10000.0)
    controller = PDController(
        kp_pos=1500.0, kd_pos=0.0,
        kp_angle=1000.0, kd_angle=8.0
    )
    
    # 初始状态：摆杆向下，小车偏离中心
    initial_state = np.array([0.5, 0.0, 0.8, 0.0])
    state = env.reset(initial_state)
    
    # 运行仿真
    for i in range(10000):
        # 计算控制输入
        u = controller.compute_control(state)
        
        # 执行一步
        state, done, _ = env.step(u)
        
        if done:
            print(f"仿真在 {i*env.dt:.2f} 秒后结束")
            break
    
    # 绘制结果
    # env.render(mode='static')
    env.render(mode='human')
    # 显示最终状态
    print(f"最终状态: {state}")
    print(f"目标角度 (π): {np.pi}")
    print(f"角度误差: {state[2] - np.pi} rad")
    
def test_ilqr_controller():
    """测试ILQR控制器"""
    print("\n测试ILQR控制器...")
    controller = PDController(
        kp_pos=15.0, kd_pos=80.0,
        kp_angle=1000.0, kd_angle=8.0
    )
    # 创建环境和控制器
    
    env = InvertedPendulumEnv(dt=0.01, max_force=10000.0)
    real_env = InvertedPendulumEnv(dt=0.01, max_force=10000.0)
    initial_state = np.array([0.5, 0.0, 0.8, 0.0])
    x_ref = np.array([0.0, 0.0, 0.0, 0.0]).reshape(-1,1)
    u_ref = 0
    N = 50
    Q = np.diag([1.0, 1.0, 100.0, 1.0])
    R = np.eye(1)* 0.01
    state = env.reset(initial_state)
    real_state = real_env.reset(initial_state)
    solver = ILQR(env,0.01,Q,R,N)
    
    initial_control = np.ones((N,1)) * 1600

    max_iter = 100
    eps = 0.00001
    regularization = 1.0
    solver.ilqr_setup(max_iter,eps,x_ref,u_ref,regularization)

    
    # 运行仿真
    for i in range(100):
        # 计算控制输入
        u = controller.compute_control(initial_state)
        # u = 0
        solver.update_state(initial_state)
        solver.ForwardPass(initial_control)
        while(not solver.done):
            expected_reduction = solver.BackwardPass()
            solver.line_search_with_expected(expected_reduction)
        state, done, _ = real_env.step(u + solver.control_trajectory[-1],'rk4')
        initial_control = solver.control_trajectory
        initial_state = state
        if done:
            print(f"仿真在 {i*env.dt:.2f} 秒后结束")
            break
    
    # 绘制结果
    # env.render(mode='static')
    real_env.render(mode='human')
    # 显示最终状态
    print(f"最终状态: {state}")
    print(f"目标角度 (π): {np.pi}")
    print(f"角度误差: {state[2] - np.pi} rad")

# 运行测试
if __name__ == "__main__":
    # test_pd_controller()
    test_ilqr_controller()

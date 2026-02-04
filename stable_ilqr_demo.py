import numpy as np
from CartPole import InvertedPendulumEnv
from ilqr import ILQR
from visualization import visualize_ilqr_results, plot_iteration_comparison

def run_stable_ilqr_example():
    """
    运行一个稳定的小例子
    """
    # 设置环境和参数
    env = InvertedPendulumEnv(dt=0.01, max_force=50.0)  # 降低最大力
    initial_state = np.array([0.05, 0.0, 0.05, 0.0])  # 很小的初始偏差
    N = 10  # 很短的时间范围
    initial_control = np.zeros((N,1))
    Q = np.eye(len(initial_state)) * 0.5  # 很小的权重
    R = np.eye(1) * 0.1
    max_iter = 5  # 很少的迭代次数
    eps = 0.5  # 很大的收敛阈值
    regularization = 1e-4  # 正则化
    dt = 0.01

    # 创建iLQR求解器
    solver = ILQR(env, initial_state, dt, Q, R, N)
    x_ref = np.array([0.0, 0.0, 0.0, 0.0]).reshape(-1,1)
    u_ref = 0
    solver.ilqr_setup(max_iter, eps, x_ref, u_ref, regularization)
    
    print("运行稳定的小例子...")
    print(f"初始状态: {initial_state}")
    print(f"参考状态: {x_ref.flatten()}")
    print(f"时间步数: {N}, 时间间隔: {dt}s")
    
    # 运行前向传播初始化
    solver.ForwardPass(initial_control)
    print(f"初始化成本: {solver.last_Vx:.4f}")
    
    # 只运行一次迭代以避免复杂问题
    try:
        solver.BackwardPass()
        solver.line_search_with_expected()
        print(f"第一次迭代后成本: {solver.last_Vx:.4f}")
    except Exception as e:
        print(f"迭代过程中发生错误: {e}")
    
    print(f"最终成本: {solver.last_Vx:.4f}")
    print(f"最终状态: {solver.state_trajectory[-1, :]}")
    
    # 进行可视化
    print("\n生成可视化图表...")
    visualize_ilqr_results(solver, show_cost=True, show_state=True, show_control=True)
    
    # 绘制迭代比较（如果记录了历史轨迹）
    print("生成迭代演化图...")
    plot_iteration_comparison(solver)


if __name__ == '__main__':
    print("稳定版iLQR求解器可视化演示")
    print("="*50)
    
    # 运行稳定示例
    run_stable_ilqr_example()
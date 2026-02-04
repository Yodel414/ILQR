import numpy as np
import matplotlib.pyplot as plt


def plot_cost_history(ilqr_solver, title="Cost Function History"):
    """
    绘制代价函数随迭代次数的变化
    
    参数:
    ilqr_solver: 已经运行过的iLQR求解器实例
    title: 图表标题
    """
    if not hasattr(ilqr_solver, 'cost_history') or len(ilqr_solver.cost_history) == 0:
        print("错误: iLQR求解器没有成本历史数据")
        return
    
    iterations = range(len(ilqr_solver.cost_history))
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, ilqr_solver.cost_history, 'b-', linewidth=2, marker='o', markersize=6)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数刻度以便更好地观察变化
    plt.tight_layout()
    plt.show()


def plot_final_trajectory(ilqr_solver, title="Final State Trajectory"):
    """
    绘制最终的状态轨迹
    
    参数:
    ilqr_solver: 已经运行过的iLQR求解器实例
    title: 图表标题
    """
    if not hasattr(ilqr_solver, 'state_trajectory'):
        print("错误: iLQR求解器没有状态轨迹数据")
        return
        
    if ilqr_solver.state_trajectory.size == 0:
        print("错误: iLQR求解器的状态轨迹为空")
        return
    
    N = ilqr_solver.N + 1  # 包含初始状态
    time_steps = np.arange(N) * ilqr_solver.dt
    
    # 获取状态维度
    state_dim = ilqr_solver.state_dim
    
    # 创建子图
    fig, axes = plt.subplots(state_dim, 1, figsize=(12, 3 * state_dim))
    
    # 如果只有一个状态变量，axes不是数组，需要特殊处理
    if state_dim == 1:
        axes = [axes]
    
    # 定义状态变量名称
    state_names = ['Position', 'Velocity', 'Angle', 'Angular Velocity']
    
    for i in range(state_dim):
        axes[i].plot(time_steps, ilqr_solver.state_trajectory[:, i], 
                    linewidth=2, label=f'State {i}')
        
        # 如果有参考轨迹，也绘制出来
        if hasattr(ilqr_solver, 'x_ref') and ilqr_solver.x_ref is not None:
            ref_value = ilqr_solver.x_ref[i, 0] if ilqr_solver.x_ref.shape[0] > i else 0
            axes[i].axhline(y=ref_value, color='r', linestyle='--', 
                           label=f'Reference {i}: {ref_value:.2f}', alpha=0.7)
        
        axes[i].set_xlabel('Time (s)')
        state_name = state_names[i] if i < len(state_names) else f'State {i}'
        axes[i].set_ylabel(state_name)
        axes[i].set_title(f'{state_name} Trajectory')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_control_trajectory(ilqr_solver, title="Control Input Trajectory"):
    """
    绘制控制输入轨迹
    
    参数:
    ilqr_solver: 已经运行过的iLQR求解器实例
    title: 图表标题
    """
    if not hasattr(ilqr_solver, 'control_trajectory'):
        print("错误: iLQR求解器没有控制轨迹数据")
        return
        
    if ilqr_solver.control_trajectory.size == 0:
        print("错误: iLQR求解器的控制轨迹为空")
        return
    
    N = ilqr_solver.N
    time_steps = np.arange(N) * ilqr_solver.dt
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, ilqr_solver.control_trajectory.flatten(), 
             linewidth=2, marker='o', markersize=4)
    plt.xlabel('Time (s)')
    plt.ylabel('Control Input')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # 如果有参考控制，也绘制出来
    if hasattr(ilqr_solver, 'u_ref') and ilqr_solver.u_ref is not None:
        plt.axhline(y=ilqr_solver.u_ref, color='r', linestyle='--', 
                   label=f'Reference Control: {ilqr_solver.u_ref:.2f}', alpha=0.7)
        plt.legend()
    
    plt.tight_layout()
    plt.show()


def visualize_ilqr_results(ilqr_solver, show_cost=True, show_state=True, show_control=True):
    """
    综合可视化iLQR求解结果
    
    参数:
    ilqr_solver: 已经运行过的iLQR求解器实例
    show_cost: 是否显示代价函数历史
    show_state: 是否显示状态轨迹
    show_control: 是否显示控制轨迹
    """
    if show_cost:
        plot_cost_history(ilqr_solver, "iLQR Cost Function History")
    
    if show_state:
        plot_final_trajectory(ilqr_solver, "iLQR Final State Trajectory")
    
    if show_control:
        plot_control_trajectory(ilqr_solver, "iLQR Control Input Trajectory")


def plot_iteration_comparison(ilqr_solver, title="State Trajectory Evolution"):
    """
    绘制不同迭代次数下的状态轨迹变化（如果记录了历史轨迹）
    
    参数:
    ilqr_solver: 已经运行过的iLQR求解器实例
    title: 图表标题
    """
    if not hasattr(ilqr_solver, 'state_trajectories_history') or len(ilqr_solver.state_trajectories_history) == 0:
        print("警告: 没有找到状态轨迹历史数据，无法绘制迭代比较图")
        return
    
    num_iterations = len(ilqr_solver.state_trajectories_history)
    state_dim = ilqr_solver.state_dim
    
    # 选择要显示的迭代（例如：每隔几轮显示一次，最多显示10条）
    max_display = min(10, num_iterations)
    display_indices = np.linspace(0, num_iterations-1, max_display, dtype=int)
    
    # 创建子图
    fig, axes = plt.subplots(state_dim, 1, figsize=(12, 3 * state_dim))
    
    # 如果只有一个状态变量，axes不是数组，需要特殊处理
    if state_dim == 1:
        axes = [axes]
    
    # 定义状态变量名称
    state_names = ['Position', 'Velocity', 'Angle', 'Angular Velocity']
    
    for idx in display_indices:
        traj = ilqr_solver.state_trajectories_history[idx]
        N = traj.shape[0]
        time_steps = np.arange(N) * ilqr_solver.dt
        
        alpha = 0.3 + 0.7 * (idx / (num_iterations - 1)) if num_iterations > 1 else 1.0
        color = plt.cm.viridis(idx / max(display_indices + [1]))  # 使用颜色渐变
        
        for i in range(state_dim):
            axes[i].plot(time_steps, traj[:, i], 
                        linewidth=1.5, alpha=alpha, color=color,
                        label=f'Iter {idx}' if i == 0 and idx in display_indices[::len(display_indices)//3] else "")
            
            # 如果是最后一次迭代，用不同样式突出显示
            if idx == num_iterations - 1:
                axes[i].plot(time_steps, traj[:, i], 
                           linewidth=3, alpha=1.0, color='red',
                           label=f'Final Iter {idx}' if i == 0 else "")
    
    for i in range(state_dim):
        axes[i].set_xlabel('Time (s)')
        state_name = state_names[i] if i < len(state_names) else f'State {i}'
        axes[i].set_ylabel(state_name)
        axes[i].set_title(f'{state_name} Trajectory Evolution')
        axes[i].grid(True, alpha=0.3)
        if i == 0:  # 只在第一个子图显示图例
            axes[i].legend(loc='upper right', fontsize='small')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
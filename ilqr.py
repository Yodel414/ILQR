import numpy as np
from CartPole import InvertedPendulumEnv
import matplotlib.pyplot as plt

class ILQR:
    def __init__(self,env:InvertedPendulumEnv,dt,Q,R,N):
        self.env = env
        self.Q = Q
        self.Qf = 0.0*Q
        self.R = R
        self.dt = dt
        self.N = N
        self.state_dim = len(Q[:,1])
        self.control_dim = len(R)
        self.state_trajectory = np.zeros((N + 1,self.state_dim))
        self.control_trajectory = np.zeros((N,1))
        self.kk_trajectory = np.zeros((N , 1))
        self.Kk_trajectory = np.zeros((N , self.state_dim))
        self.done = False

        # 添加用于可视化的数据存储
        self.cost_history = []  # 存储每次迭代的成本
        self.state_trajectories_history = []  # 存储每次迭代的状态轨迹
        self.control_trajectories_history = []  # 存储每次迭代的控制轨迹
        self.Q_uu_list_history = []  # 存储每次迭代的Q_uu矩阵
    def ilqr_setup(self,max_iter,eps,x_ref,u_ref,regularization):
        self.max_iter = max_iter
        self.eps = eps
        self.x_ref = x_ref
        self.u_ref = u_ref
        self.iter = 0
        self.last_Vx = 0
        self.regularization = regularization
    def update_state(self,initial_state):
        self.initial_state = initial_state
    def ForwardPass(self,u):
        xk_new = self.initial_state.reshape(-1)
        self.state_trajectory[0,:] =xk_new
        Vx = 0
        for index in range(0,self.N):
            uk = u[index]
            delta_x = xk_new.reshape(-1,1) - self.x_ref
            delta_u = uk - self.u_ref
            Vx =  Vx + 0.5 * delta_x.T @ self.Q @ delta_x \
                + 0.5 * delta_u.T @ self.R @ delta_u
            state_new, done, _  = self.env.step(uk)
            self.state_trajectory[index + 1,:] = state_new.flatten()
            self.control_trajectory[index,:] = uk
            xk_new = self.state_trajectory[index + 1,:]
        delta_x = xk_new.reshape(-1,1) - self.x_ref
        self.last_Vx = Vx  + 0.5 * delta_x.T @ self.Q @ delta_x

        # 记录当前迭代的状态和控制轨迹用于可视化
        self.state_trajectories_history.append(self.state_trajectory.copy())
        
    def line_search_with_expected(self,expected_reduction):
        if abs(expected_reduction) > 0.1:
        # 期望下降较大，可以尝试更大步长
            alphas = [1.0, 0.8, 0.6, 0.4, 0.2]
        else:
            # 期望下降较小，使用更保守的步长
           alphas = [0.5, 0.3, 0.1, 0.05]
        c = 1e-4
        best_alpha = 0.0
        best_states = self.state_trajectory
        best_controls =self.control_trajectory
        best_cost = self.last_Vx
        # best_du =self.du
        armijo_condition = True
        # best_du
        for alpha in alphas:
            cur_traj,cur_control,cur_vx,du = self.ForwardPassWithFeedback(alpha)
            actual_reduction = self.last_Vx - cur_vx
            if expected_reduction > 0:
                armijo_condition = actual_reduction >= c * alpha * expected_reduction
            else:
                armijo_condition = actual_reduction > 0
            if cur_vx < best_cost:
                best_alpha = alpha
                best_states = cur_traj
                best_controls = cur_control
                best_cost = cur_vx
                best_du = du
        # self.du = best_du
        self.state_trajectory = best_states
        self.control_trajectory = best_controls
                # 记录成本历史用于可视化
        self.cost_history.append(best_cost)
        self.iter = self.iter + 1
        self.control_trajectories_history.append(self.control_trajectory.copy())
        cur_eps = best_cost - self.last_Vx
        if self.iter > self.max_iter or abs(cur_eps)<self.eps :
        # if self.iter > self.max_iter:
            self.done = True
        else:
            self.last_Vx = best_cost
    def ForwardPassWithFeedback(self,alpha):
        xk_new = self.initial_state.reshape(-1,1)
        tmp_traj = np.zeros_like(self.state_trajectory)
        tmp_control = np.zeros_like(self.control_trajectory)
        tmp_traj[0, :] = xk_new.flatten()
        du_list = []
        vx = 0
        old_state_traj = self.state_trajectory.copy()
        old_ctrl_traj = self.control_trajectory.copy()
        self.env.reset(xk_new)
        for index in range(0,self.N):
            uk = old_ctrl_traj[index,:]
            kk = self.kk_trajectory[index,:]
            Kk = self.Kk_trajectory[index,:]
            
            du = alpha * kk + Kk @ (xk_new - old_state_traj[index,:].reshape(-1,1))
            uk_new = uk + du
            du_list.append(du)
            state_new, done, _  = self.env.step(uk_new)
            tmp_traj[index + 1,:] = state_new.flatten()
            tmp_control[index,:] = uk_new
            xk_new = tmp_traj[index + 1,:].reshape(-1,1)
            delta_x = xk_new  - self.x_ref
            delta_u = uk_new - self.u_ref
            vx =  vx + 0.5 * delta_x.T @ self.Q @ delta_x \
                + 0.5 * delta_u.T @ self.R @ delta_u
        delta_x = xk_new.reshape(-1,1) - self.x_ref
        vx = vx  + 0.5 * delta_x.T @ self.Qf @ delta_x
        return tmp_traj,tmp_control,vx,du_list

    def BackwardPass(self):
        X_N = self.state_trajectory[-1,:].reshape(-1,1).copy()
        self.CostDerivatives(X_N,0,True)
        V_k_1_x = self.lfx
        V_k_1_xx = self.lfxx
        # V_k_1_xx = 0.5 * (V_k_1_xx + V_k_1_xx.T)
        Q_uu_list = []
        regularization = self.regularization
        expected_reduction = 0
        for index in range(self.N - 1,-1,-1):
            xk = self.state_trajectory[index,:].reshape(-1,1).copy()
            uk = self.control_trajectory[index,:].copy()
            if index ==  N - 1:
                print(1)
            fx,fu = self.env.GetDerivatives(xk,uk)
            self.CostDerivatives(xk,uk,False)
            Qx = self.lx + fx.T @ V_k_1_x
            Qu = self.lu + fu.T @ V_k_1_x
            Qxx = self.lxx + fx.T @ V_k_1_xx @ fx
            Quu = self.luu + fu.T @ V_k_1_xx @ fu
            Qux = self.lux + fu.T @ V_k_1_xx @ fx
            Q_uu_list.insert(0, Quu.copy())
            Quu_reg = Quu + regularization * np.eye(self.control_dim)

            cond_number = np.linalg.cond(Quu_reg)
            # if cond_number > 1e10:
            #     print(f"警告：Q_uu条件数过大: {cond_number:.2e}")
            #     # 可以增加正则化
            #     regularization *= 2
            #     Quu_reg = Quu + regularization * np.eye(self.control_dim)
            try:
                Quu_inv = np.linalg.inv(Quu_reg)
            except np.linalg.LinAlgError:
                # 如果求逆失败，增加正则化
                regularization *= 10
                Quu_reg = Quu + regularization * np.eye(self.control_dim)
                Quu_inv = np.linalg.inv(Quu_reg)
            kk = -Quu_inv @ Qu
            Kk = -Quu_inv @ Qux
            V_k_1_x = Qx + np.transpose(Kk) @ Quu_reg @ kk + np.transpose(Kk) @ Qu +\
                np.transpose(Qux) @ kk
            V_k_1_xx = Qxx + np.transpose(Kk) @ Quu_reg @ Kk + np.transpose(Kk) @ Qux +\
                np.transpose(Qux) @ Kk
            V_k_1_xx = 0.5 * (V_k_1_xx + V_k_1_xx.T)
            self.kk_trajectory[index,:] = kk
            self.Kk_trajectory[index,:] = Kk
        self.current_regularization = regularization
        self.Q_uu_list = Q_uu_list
        # 记录Q_uu_list用于可视化
        self.Q_uu_list_history.append(Q_uu_list.copy())
        expected_reduction += -kk.T @ Qu - 0.5 * kk.T @ Quu @ kk
        return expected_reduction
                 
    def CostDerivatives(self,x, u,terminal=False) :
        x_state = x.reshape(-1,1)
        u_ref = 0
        if not terminal:
            self.lx = self.Q @ (x_state - self.x_ref)  
            self.lu = self.R @ (u - u_ref)
            self.lxx = self.Q
            self.luu = self.R
            self.lux = np.zeros((1,len(self.x_ref)))
        else:
            self.lfx = self.Qf @ (x_state - self.x_ref) 
            self.lfxx = self.Qf
    def compute_expected_reduction(self):
        expected_reduction = 0
        for kk, Quu in zip(self.kk_trajectory, self.Q_uu_list):
            term = 0.5 * kk.T @ Quu @ kk
            expected_reduction += term.item()

        return expected_reduction

    def visualize_solution_process(self):
        """可视化iLQR求解过程"""
        if not self.cost_history:
            print("没有找到成本历史数据，无法可视化求解过程")
            return

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        fig.suptitle('iLQR Solver Visualization - Solution Process', fontsize=16)

        # 1. 成本收敛曲线
        iterations = range(len(self.cost_history))
        cost_history = [item[0][0] for item in self.cost_history]
        axes[0, 0].plot(iterations, cost_history, 'b-o', markersize=4)
        axes[0, 0].set_title('Cost Convergence Over Iterations')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Total Cost')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 最终状态轨迹
        if self.state_trajectories_history:
            final_state_traj = self.state_trajectories_history[-1]
            time_steps = np.arange(final_state_traj.shape[0]) * self.dt
            for i in range(final_state_traj.shape[1]):
                axes[0, 1].plot(time_steps, final_state_traj[:, i], label=f'State {i+1}', linewidth=2)
            axes[0, 1].set_title('Final State Trajectory')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('State Value')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. 最终控制输入轨迹
        if self.control_trajectories_history:
            final_control_traj = self.control_trajectories_history[-1]
            control_time_steps = np.arange(final_control_traj.shape[0]) * self.dt
            axes[1, 0].plot(control_time_steps, final_control_traj.flatten(), 'm-', linewidth=2)
            axes[1, 0].set_title('Final Control Input Trajectory')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Control Input')
            axes[1, 0].grid(True, alpha=0.3)

        # 4. 收敛细节
        if len(self.cost_history) > 1:
            cost_diff = np.diff(cost_history)
            axes[1, 1].plot(range(1, len(cost_diff)+1), np.abs(cost_diff), 'r-o', markersize=4)
            axes[1, 1].set_title('Absolute Cost Change Per Iteration')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('|Cost Change|')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def visualize_final_solution(self):
        """可视化最终解决方案"""
        if not self.state_trajectory.size or not self.control_trajectory.size:
            print("没有找到最终解，无法可视化")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('iLQR Solver - Final Solution Visualization', fontsize=16)

        # 状态轨迹
        time_steps = np.arange(self.state_trajectory.shape[0]) * self.dt
        for i in range(self.state_trajectory.shape[1]):
            axes[0, 0].plot(time_steps, self.state_trajectory[:, i], label=f'State {i+1}', linewidth=2)
        axes[0, 0].set_title('Final State Trajectory')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('State Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 控制输入
        control_time_steps = np.arange(self.control_trajectory.shape[0]) * self.dt
        axes[0, 1].plot(control_time_steps, self.control_trajectory.flatten(), 'm-', linewidth=2)
        axes[0, 1].set_title('Final Control Input')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Control Input')
        axes[0, 1].grid(True, alpha=0.3)

        # 状态相位图（例如，位置vs速度，角度vs角速度）
        if self.state_trajectory.shape[1] >= 4:
            axes[1, 0].plot(self.state_trajectory[:, 0], self.state_trajectory[:, 1], 'b-', linewidth=2)
            axes[1, 0].plot(self.state_trajectory[0, 0], self.state_trajectory[0, 1], 'go', markersize=8, label='Start')
            axes[1, 0].plot(self.state_trajectory[-1, 0], self.state_trajectory[-1, 1], 'ro', markersize=8, label='End')
            axes[1, 0].set_title('Phase Plot: Position vs Velocity')
            axes[1, 0].set_xlabel('Position')
            axes[1, 0].set_ylabel('Velocity')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].plot(self.state_trajectory[:, 2], self.state_trajectory[:, 3], 'r-', linewidth=2)
            axes[1, 1].plot(self.state_trajectory[0, 2], self.state_trajectory[0, 3], 'go', markersize=8, label='Start')
            axes[1, 1].plot(self.state_trajectory[-1, 2], self.state_trajectory[-1, 3], 'ro', markersize=8, label='End')
            axes[1, 1].set_title('Phase Plot: Angle vs Angular Velocity')
            axes[1, 1].set_xlabel('Angle (rad)')
            axes[1, 1].set_ylabel('Angular Velocity (rad/s)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
if __name__ == '__main__':
    env = InvertedPendulumEnv(dt=0.01, max_force=10000.0)
    
    initial_state = np.array([0.5, 0.0, 0.3, 0.0]).reshape(-1,1)
    state = env.reset(initial_state)
    N = 18
    initial_control = np.ones((N,1)) * 1200.0
    Q = np.diag([1.0, 0.10, 10.0, 0.010]) * 100
    R = np.eye(1)* 0.01
    max_iter = 1000
    eps = 0.01
    regularization = 0.01
    solver = ILQR(env,0.1,Q,R,N)
    x_ref = np.array([0.0, 0.0, 0.0, 0.0]).reshape(-1,1)
    u_ref = 0
    solver.ilqr_setup(max_iter,eps,x_ref,u_ref,regularization)
    solver.update_state(initial_state)
    solver.ForwardPass(initial_control)
    while(not solver.done):
        expected_reduction = solver.BackwardPass()
        solver.line_search_with_expected(expected_reduction)
    print(f"iLQR converged in {solver.iter} iterations")

    # 调用可视化功能
    solver.visualize_solution_process()  # 显示求解过程
    solver.visualize_final_solution()    # 显示最终解

    plt.show()      
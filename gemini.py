import numpy as np
import matplotlib.pyplot as plt

class ILQR:
    def __init__(self, dynamics_func, cost_func, state_dim, action_dim, dt=0.01):
        """
        标准 iLQR 控制器 Python 实现
        """
        self.f = dynamics_func
        self.cost = cost_func
        self.dx = state_dim
        self.du = action_dim
        self.dt = dt
        
        # 正则化参数 (Levenberg-Marquardt)
        self.mu = 1.0
        self.mu_min = 1e-6
        self.mu_max = 1e10
        self.delta_0 = 2.0
        self.delta = self.delta_0

    def get_derivatives(self, x, u):
        """使用有限差分计算导数"""
        eps = 1e-5
        
        # 动力学 Jacobian
        fx = np.zeros((self.dx, self.dx))
        fu = np.zeros((self.dx, self.du))
        for i in range(self.dx):
            dx_vec = np.zeros(self.dx); dx_vec[i] = eps
            fx[:, i] = (self.f(x + dx_vec, u) - self.f(x - dx_vec, u)) / (2 * eps)
        for i in range(self.du):
            du_vec = np.zeros(self.du); du_vec[i] = eps
            fu[:, i] = (self.f(x, u + du_vec) - self.f(x, u - du_vec)) / (2 * eps)

        # 成本函数导数 (简化 Hessian 近似)
        lx = np.zeros(self.dx)
        lu = np.zeros(self.du)
        for i in range(self.dx):
            dx_vec = np.zeros(self.dx); dx_vec[i] = eps
            lx[i] = (self.cost(x + dx_vec, u,False) - self.cost(x - dx_vec, u,False)) / (2 * eps)
        for i in range(self.du):
            du_vec = np.zeros(self.du); du_vec[i] = eps
            lu[i] = (self.cost(x, u + du_vec,False) - self.cost(x, u - du_vec,False)) / (2 * eps)
        
        lxx = np.eye(self.dx) * 0.1 # 实际应用中建议使用更精确的 Hessian
        luu = np.eye(self.du) * 0.1
        lux = np.zeros((self.du, self.dx))

        return fx, fu, lx, lu, lxx, luu, lux

    def forward_rollout(self, x0, U, K=None, d=None, X_ref=None, alpha=1.0):
        """执行轨迹演化"""
        N = U.shape[0]
        X = np.zeros((N + 1, self.dx))
        U_new = np.zeros((N, self.du))
        X[0] = x0
        total_cost = 0

        for k in range(N):
            if K is not None:
                delta_x = X[k] - X_ref[k]
                U_new[k] = U[k] + K[k] @ delta_x + alpha * d[k]
            else:
                U_new[k] = U[k]

            X[k+1] = self.f(X[k], U_new[k])
            total_cost += self.cost(X[k], U_new[k], False)
        
        total_cost += self.cost(X[-1], np.zeros(self.du), True)
        return X, U_new, total_cost

    def backward_pass(self, X, U):
        """反向传播计算增益"""
        N = U.shape[0]
        K = np.zeros((N, self.du, self.dx))
        d = np.zeros((N, self.du))
        
        # 终端状态导数
        vx = np.zeros(self.dx)
        eps = 1e-5
        for i in range(self.dx):
            dx_vec = np.zeros(self.dx); dx_vec[i] = eps
            vx[i] = (self.cost(X[-1] + dx_vec, None, True) - self.cost(X[-1] - dx_vec, None, True)) / (2 * eps)
        vxx = np.eye(self.dx) * 10.0

        for k in range(N - 1, -1, -1):
            fx, fu, lx, lu, lxx, luu, lux = self.get_derivatives(X[k], U[k])
            
            Qx = lx + fx.T @ vx
            Qu = lu + fu.T @ vx
            Qxx = lxx + fx.T @ vxx @ fx
            Quu = luu + fu.T @ vxx @ fu
            Qux = lux + fu.T @ vxx @ fx
            
            # 正则化
            Quu_reg = Quu + np.eye(self.du) * self.mu
            
            try:
                inv_Quu = np.linalg.inv(Quu_reg)
                d[k] = -inv_Quu @ Qu
                K[k] = -inv_Quu @ Qux
            except np.linalg.LinAlgError:
                return None, None, False

            vx = Qx + K[k].T @ Quu @ d[k] + K[k].T @ Qu + Qux.T @ d[k]
            vxx = Qxx + K[k].T @ Quu @ K[k] + K[k].T @ Qux + Qux.T @ K[k]
            vxx = 0.5 * (vxx + vxx.T)

        return K, d, True

    def solve(self, x0, U_init, max_iter=30):
        U = U_init.copy()
        X, _, cost = self.forward_rollout(x0, U)
        
        for i in range(max_iter):
            K, d, success = self.backward_pass(X, U)
            if not success: break

            # 线搜索
            alpha = 1.0
            accepted = False
            while alpha > 1e-4:
                X_new, U_new, cost_new = self.forward_rollout(x0, U, K, d, X, alpha)
                if cost_new < cost:
                    cost, X, U = cost_new, X_new, U_new
                    accepted = True
                    self.mu = max(self.mu_min, self.mu / self.delta_0)
                    break
                alpha *= 0.5
            
            if not accepted:
                self.mu *= self.delta_0
                if self.mu > self.mu_max: break
            
            print(f"Iter {i}: Cost = {cost:.4f}, mu = {self.mu:.1e}")
        return X, U

# --- 倒立摆模型 ---
def pendulum_dynamics(x, u):
    g, l, m, dt = 9.81, 1.0, 1.0, 0.05
    theta, theta_dot = x[0], x[1]
    torque = np.clip(u[0], -10, 10)
    new_theta_dot = theta_dot + dt * (3*g/(2*l) * np.sin(theta) + 3.0/(m*l**2) * torque)
    new_theta = theta + new_theta_dot * dt
    return np.array([new_theta, new_theta_dot])

def pendulum_cost(x, u, is_terminal):
    target = np.array([np.pi, 0.0]) # 目标：直立 (向上为PI)
    err = x - target
    if is_terminal:
        return 0.5 * (500 * err[0]**2 + 100 * err[1]**2)
    return 0.5 * (1.0 * err[0]**2 + 0.1 * err[1]**2 + 0.01 * u[0]**2)

if __name__ == "__main__":
    # 初始状态：静止向下 (0, 0)
    x0 = np.array([0.0, 0.0])
    horizon = 60
    u_init = np.zeros((horizon, 1))
    
    solver = ILQR(pendulum_dynamics, pendulum_cost, 2, 1)
    X, U = solver.solve(x0, u_init)

    plt.figure(figsize=(10, 4))
    plt.subplot(121); plt.plot(X[:, 0]); plt.title("Angle (Theta)"); plt.axhline(np.pi, c='r')
    plt.subplot(122); plt.step(range(horizon), U[:, 0]); plt.title("Control Torque")
    plt.show()
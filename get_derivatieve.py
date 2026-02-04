import sympy as sp
import numpy as np
# 定义符号

M, m, l, g, t = sp.symbols('M m l g t')
p, v, theta, omega, u = sp.symbols('p v theta omega u')

# 定义非线性动力学 (基于拉格朗日力学推导)
# x = [p, v, theta, omega]
# f1 = p_dot = v
# f2 = v_dot = acc_p
# f3 = theta_dot = omega
# f4 = omega_dot = acc_theta

total_mass = M + m
pendulum_moment_arm = m * l



denom = M + m * sp.sin(theta)**2

acc_p = (u + m * sp.sin(theta) * (l * omega**2 + g * sp.cos(theta))) / denom
acc_theta = (-u * sp.cos(theta) - (M + m) * g * sp.sin(theta) - m * l * omega**2 * sp.sin(theta) * sp.cos(theta)) / (l * denom)

f = sp.Matrix([v, acc_p, omega, acc_theta])
state = sp.Matrix([p, v, theta, omega])
control = sp.Matrix([u])

# 计算雅可比矩阵
Fx_analytical = f.jacobian(state)
Fu_analytical = f.jacobian(control)

# 打印其中一个复杂的项看看 (例如 d(acc_theta) / d(theta))
for i in range(4):
    for j in range(4):
        print("Fx_analytical[",i,j,"]的解析表达式：")
        sp.pprint(Fx_analytical[i,j])

for j in range(4):
    print("Fu_analytical[",j,0,"]的解析表达式：")
    sp.pprint(Fu_analytical[j])

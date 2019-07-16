import numpy as np
from mpc_optimizer import mpc_opt
import matplotlib.pyplot as plt

# Model Predictive Control
#   (x_N-x_r)^T P (x_N-x_r) + \sum_{k=0}^{N-1} (x_k-x_r)^T Q (x_k-x_r) + u_k^T R u_k \\
#   subject to & x_{k+1} = A x_k + B u_k \\
#   x_{min} <= x_k  <= x_{max} \\
#   u_{min} <= u_k  <= u_{max} \\
#   x_0 = x_bar


if __name__ == '__main__':
    Q = 5 * np.diag((0.5, 1, 0))  # state penalty
    P = 2 * np.diag((5, 5, 5))  # terminal state penalty
    R = np.eye(2)   # input penalty

    # x_dot = A x + B u
    A = np.array([[-0.79, -0.3, -0.1], [0.5, 0.82, 1.23], [0.52, -0.3, -0.5]])  # linearized system dynamics matrix
    B = np.array([[-2.04, -0.21], [-1.28, 2.75], [0.29, -1.41]])   # input matrix
    C = np.diag((0, 1, 0))  # output matrix as in y = C x

    t = np.linspace(0, 1, 50)  # sampling time

    ul, uh, xl, xh = np.array([[-2.], [-3.]]), np.array([[2.], [3.]]), np.array(
        [[-10.], [-9.], [-8.]]), np.array([[10.], [9.], [8.]])   # input and state lower and upper limits
    N = 3  # Prediction horizon (control horizon is assumed to be the same)
    ref_traj = np.array([[0], [2], [0]])
    x0, u0, = np.array([[0.0], [0.0], [0.0]]), np.array([[0.4], [0.2]])  # initial state and input guess

    mpc = mpc_opt(Q=Q, P=P, R=R, A=A, B=B, C=C, time=t, ul=ul, uh=uh, xl=xl, xh=xh, N=N, ref_traj=ref_traj)
    X, U = mpc.get_state_and_input(u0, x0)

    plt.figure(1)
    plt.plot(X.transpose())
    plt.xlabel('time')
    plt.title('states')

    plt.figure(2)
    plt.plot(U.transpose())
    plt.title('Input')
    plt.xlabel('time')

    plt.show()

    print('hi')
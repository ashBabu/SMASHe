import numpy as np
import sympy as sp
import scipy.optimize as opt
from mechanics import dynamics, kinematics
import matplotlib.pyplot as plt

# Model Predictive Control : Currently, the prediction horizon and control horizon are set the same with N
#   (x_N-x_r)^T P (x_N-x_r) + \sum_{k=0}^{N-1} (x_k-x_r)^T Q (x_k-x_r) + u_k^T R u_k \\
#   subject to & x_{k+1} = A x_k + B u_k \\
#   x_{min} <= x_k  <= x_{max} \\
#   u_{min} <= u_k  <= u_{max} \\
#   x_0 = x_bar

# TODO:
# 1. Code for adding control horizon
# 2. Code for adding u and x bounds


class mpc_opt():

    def __init__(self):
        self.optCurve, self.costs = [], []
        self.omega = 0.5
        # self.Q = 0.15*sp.eye(3)
        # self.R = 0.5*sp.eye(2)
        # self.A = sp.Matrix([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        # self.B = sp.Matrix([[0, 0], [0, 0], [1, 0], [0, 1]])
        self.Q = 5*np.diag((0.5, 1, 0))
        # self.P = 2*np.diag((0.5, 1, 0))   # terminal state penalty
        self.P = 2*np.diag((5, 5, 5))   # terminal state penalty
        self.R = np.eye(2)
        # self.A = sp.Matrix([[-0.79, -0.3, -0.1],[0.5, 0.82, 1.23], [0.52, -0.3, -0.5]])
        # self.B = sp.Matrix([[-2.04, -0.21], [-1.28, 2.75], [0.29, -1.41]])

        self.A = np.array([[-0.79, -0.3, -0.1], [0.5, 0.82, 1.23], [0.52, -0.3, -0.5]])
        self.B = np.array([[-2.04, -0.21], [-1.28, 2.75], [0.29, -1.41]])
        self.C = np.diag((0, 1, 0))

        # self.B = np.array([[-2.04, -0.21], [-1.28, 2.75], [0.29, -1.41]])
        # self.B = 3*np.eye(3, 2)
        # self.A = 2*np.eye(3)

        self.t = np.linspace(0, 1, 50)

        self.dyn = dynamics()
        self.kin = kinematics()

    def plant_model(self, x, u):
        x_dot = self.A * x + self.B * u
        return x_dot

    def transfer_matrices(self, N):
        nx, nB = self.B.shape
        Su = np.zeros((N*nx, N*nB))
        Sx = self.A
        An = self.A @ self.A
        for i in range(N-1):
            Sx = np.concatenate((Sx, An), axis=0)
            An = An @ self.A
        B = self.B
        for i in range(N):
            Su = Su + np.kron(np.eye(N, k=-i), B)
            B = self.A @ B
        return Sx, Su

    def penalties(self, N):
        return np.kron(np.eye(N-1), self.Q), np.kron(np.eye(N), self.R), np.kron(np.eye(N), self.C)

    def plant(self, Sx, Su, x0, u):
        x = Sx @ x0 + Su @ u
        return x

    def block_diag(self, Q, P):
        nx, nu = self.B.shape
        t1 = np.zeros((N * nx, N * nx))
        t1[0:(N - 1) * nx, 0:(N - 1) * nx] = Q
        t2 = np.zeros((N, N))
        t2[-1, -1] = 1
        t3 = np.kron(t2, P)
        return t1 + t3  # diag([Q, Q, ....., P])

    def cost_function(self, u, *args):
        x0, t, N = args
        u = u.reshape(len(u), -1)
        Sx, Su = self.transfer_matrices(N)
        Qs, Rs, Cs = self.penalties(N)
        Q_blk = self.block_diag(Qs, self.P)
        x_N = self.plant(Sx, Su, x0, u)
        y = Cs @ x_N
        y_ref = self.ref_trajectory(t)
        y_ref_lifted = np.tile(y_ref, (N, 1))
        error = y_ref_lifted - y
        cost = error.transpose() @ Q_blk @ error + u.transpose() @ Rs @ u + x0.transpose() @ self.Q @ x0
        return cost

    def end_effec_pose(self, q):
        pos, _ = self.kin.fwd_kin_numeric(self.kin.l, q)
        return pos

    # @staticmethod
    def ref_trajectory(self, i):  # y = 3*sin(2*pi*omega*t)
        # y_ref = 3 * np.sin(2*np.pi*self.omega*self.t[i])
        y_ref = np.array([[0], [2], [0]])
        return y_ref
        # return sp.Matrix(([[self.t[i]], [y], [0]]))

    def cost_function_old(self, u, *args):
        x0, t = args
        u = u.reshape(len(u), -1)
        x0 = x0.reshape(len(x0), -1)
        x = self.A @ x0 + self.B @ u
        y = self.C @ x
        # q = [x1[0], x1[1]]
        # pos = self.end_effec_pose(q)
        y_ref = self.ref_trajectory(t)
        error = y - y_ref
        cost = error.transpose() @ self.Q @ error + u.transpose() @ self.R @ u
        return cost

    def cost_gradient(self, u, *args):
        x0, t, N = args
        u = u.reshape(len(u), -1)
        Sx, Su = self.transfer_matrices(N)
        Qs, Rs, Cs = self.penalties(N)
        Q_blk = self.block_diag(Qs, self.P)
        x_N = self.plant(Sx, Su, x0, u)
        y = Cs @ x_N
        y_ref = self.ref_trajectory(t)
        y_ref_lifted = np.tile(y_ref, (N, 1))
        error = y_ref_lifted - y
        cost_gradient = np.vstack((Q_blk @ error, Rs @ u, self.Q @ x0))
        return cost_gradient

    def constraints(self, u, *args):
        def x_bounds(x):
            return x.sum() - 1

        # def dot_constraint(x):
        #     return x.dot(R) - E

        # def objective(x):
        #     return x.dot(M).dot(x)

        umin = np.array([-2., -3.])
        umax = np.array([2., 3.])
        xmin = np.array([-10., -9., -8.])
        xmax = np.array([10., 9., 8.])
        lb = sp.Matrix([xmin, umin])
        ub = sp.Matrix([xmax, umax])

        bounds = [(lb, ub)]
        # constraints = (
        #     dict(type='eq', fun=simplex_constraint),
        #     dict(type='eq', fun=dot_constraint))

    def optimise(self, u0, x0, t, N=10, k=1):
        umin = [-2., -3.]
        umax = [2., 3.]
        xmin = [-10., -9., -8.]
        xmax = [10., 9., 8.]
        # bounds = ((xmin[0], xmax[0]), (xmin[1], xmax[1]), (xmin[2], xmax[2]), (umin[0], umax[0]), (umin[1], umax[1]))
        bounds1 = ((umin[0], umax[0]), (umin[1], umax[1]))

        # U = opt.minimize(self.cost_function, u0, args=(x0, t, N), method='SLSQP', bounds=bounds1, jac=self.cost_gradient(u0, x0, t, N),
        #                  options={'maxiter': 200, 'disp': True})

        U = opt.minimize(self.cost_function, u0, args=(x0, t, N), method='SLSQP', options={'maxiter': 200, 'disp': True})
        U = U.x
        return U


if __name__ == '__main__':
    mpc = mpc_opt()
    # P, H = mpc.imgpc_predmat(mpc.A, mpc.B, np.eye(3), 0, 5)
    pos = sp.zeros(3, len(mpc.t))
    x0, u0, = np.array([[0.0], [0.0], [0.0]]), np.array([[0.4], [0.2]])
    X, U = np.zeros((len(x0), len(mpc.t))), np.zeros((len(u0), len(mpc.t)))
    nx, nu = mpc.B.shape

    # ######## loop for cost_function ###########
    N = 2   # Prediction horizon
    u0 = np.tile(u0, (N, 1))

    for i in range(len(mpc.t)):
        print('i = :', i)
        U[:, i], X[:, i] = u0[0:nu].transpose(), x0.transpose()
        u = mpc.optimise(u0, x0, i, N=N, k=2)
        u0 = u
        u = u[0:nu].reshape(nu, -1)
        x0 = x0.reshape(len(x0), -1)
        x0 = mpc.A @ x0 + mpc.B @ u

    plt.figure(1)
    plt.plot(X[0, :], '--r')
    plt.plot(X[1, :], '--b')
    plt.plot(X[2, :], '--g')
    # plt.ylim(-2, 2)
    plt.xlabel('time')
    plt.title('states')

    plt.figure(2)
    plt.plot(U[0, :], '--r')
    plt.plot(U[1, :], '--b')
    # plt.ylim(-1, 1)
    plt.title('Input')
    plt.xlabel('time')

    plt.show()

    print('hi')
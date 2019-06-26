import numpy as np
import sympy as sp
import scipy.optimize as opt
from scipy.linalg import block_diag
from mechanics import dynamics, kinematics
import matplotlib.pyplot as plt

# Model Predictive Control : Currently, the prediction and control horizon are set the same with N
#   (x_N-x_r)^T P (x_N-x_r) + \sum_{k=0}^{N-1} (x_k-x_r)^T Q (x_k-x_r) + u_k^T R u_k \\
#   subject to & x_{k+1} = A x_k + B u_k \\
#   x_{min} <= x_k  <= x_{max} \\
#   u_{min} <= u_k  <= u_{max} \\
#   x_0 = x_bar

# TODO:
# 1. Code for adding control horizon


class mpc_opt():

    def __init__(self, A=None, B=None, C=None, Q=None, R=None, P=None,
                 xl=None, xh=None, ul=None, uh=None, N=4, x0=None, time=None):

        if not A.any():
            self.Q = 5*np.diag((0.5, 1, 0))
            self.P = 2*np.diag((5, 5, 5))   # terminal state penalty
            self.R = np.eye(2)

            self.A = np.array([[-0.79, -0.3, -0.1], [0.5, 0.82, 1.23], [0.52, -0.3, -0.5]])
            self.B = np.array([[-2.04, -0.21], [-1.28, 2.75], [0.29, -1.41]])
            self.C = np.diag((0, 1, 0))

            self.t = np.linspace(0, 1, 50)

            self.ul, self.uh, self.xl, self.xh = np.array([[-2.], [-3.]]), np.array([[2.], [3.]]), np.array([[-10.], [-9.], [-8.]]), np.array([[10.], [9.], [8.]])
            self.x0 = np.array([[0.0], [0.0], [0.0]])
            self.N = 3   # # Prediction horizon

        else:
            self.Q = Q  # state penalty
            self.P = P  # terminal state penalty
            self.R = R  # input penalty

            # x_dot = A x + B u
            self.A = A  # linear system dynamics matrix
            self.B = B  # input matrix
            self.C = C  # output matrix

            self.t = time

            self.ul, self.uh, self.xl, self.xh = ul, uh, xl, xh
            self.x0 = x0
            self.N = N  # # Prediction horizon

        self.optCurve, self.costs = [], []
        self.omega = 0.5
        self.dyn = dynamics()
        self.kin = kinematics()

    def plant_model(self, x, u):
        x_dot = self.A * x + self.B * u
        return x_dot

    def transfer_matrices(self,):
        N = self.N
        nx, nB = self.B.shape
        Su = np.zeros((N*nx, N*nB))
        Sx = self.C @ self.A
        An = self.A @ self.A
        for i in range(N-1):
            Sx = np.concatenate((Sx, self.C @ An), axis=0)
            An = An @ self.A
        B = self.B
        for i in range(N):
            Su = Su + np.kron(np.eye(N, k=-i), self.C @ B)
            B = self.A @ B
        return Sx, Su

    def penalties(self, ):
        N = self.N
        if N == 1:
            Qs = self.Q
        else:
            Qs = np.kron(np.eye(N - 1), self.Q)
        return Qs, np.kron(np.eye(N), self.R), np.kron(np.eye(N), self.C)

    def plant(self, Sx, Su, x0, u):
        x = Sx @ x0 + Su @ u
        return x

    def block_diag(self, Q, P):
        N = self.N
        nx, nu = self.B.shape
        t1 = np.zeros((N * nx, N * nx))
        if N != 1:
            t1[0:(N - 1) * nx, 0:(N - 1) * nx] = Q
        else:
            t1 = Q
        t2 = np.zeros((N, N))
        t2[-1, -1] = 1
        t3 = np.kron(t2, P)
        return t1 + t3  # diag([Q, Q, ....., P])

    def cost_function(self, u, *args):
        x0, t, = args
        N = self.N
        u = u.reshape(len(u), -1)
        Sx, Su = self.transfer_matrices()
        Qs, Rs, Cs = self.penalties()
        Q_blk = self.block_diag(Qs, self.P)
        G = 2 * (Rs + Su.transpose() @ Q_blk @ Su)
        F = 2 * (Su.transpose() @ Q_blk @ Sx)
        K = 2 * Su.transpose() @ Q_blk
        y_ref = self.ref_trajectory(t)
        y_ref_lifted = np.tile(y_ref, (N, 1))
        cost = 0.5 * u.transpose() @ G @ u + u.transpose() @ F @ x0 - u.transpose() @ K @ y_ref_lifted
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
        N = self.N
        x0, t = args
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
        x0, t = args
        N = self.N
        nx, nu = self.B.shape
        Ix, Iu = np.eye(nx), np.eye(nu)
        zx, zu = np.zeros((2 * nu, nx)), np.zeros((2 * nx, nu))
        Mi = np.vstack((zx, -Ix, Ix))
        Ei = np.vstack((-Iu, Iu, zu))
        bi = np.vstack((-self.ul, self.uh, -self.xl, self.xh))
        MN = np.vstack((-Ix, Ix))
        bN = np.vstack((-self.xl, self.xh))
        c = np.vstack((np.tile(bi, (N, 1)), bN))
        tp = c.shape[0]
        D = np.vstack((Mi, np.zeros((tp - Mi.shape[0], Mi.shape[1]))))
        if N == 1:
            b = Mi
        else:
            a = np.kron(np.eye(N - 1), Mi)
            b = block_diag(a, MN)
        tt = c.shape[0] - b.shape[0]
        q = np.zeros((tt, nx * N))
        M = np.vstack((q, b))
        aa = np.kron(np.eye(N), Ei)
        tt1 = c.shape[0] - aa.shape[0]
        qq = np.zeros((tt1, nu * N))
        Eps = np.vstack((aa, qq))
        Sx, Su = self.transfer_matrices()
        L = M @ Su + Eps
        W = -D - M @ Sx
        u = u.reshape(len(u), -1)
        con_ieq = c + W @ x0 - L @ u
        return np.squeeze(con_ieq)

    def optimise(self, u0, x0, t):
        con_ineq = {'type': 'ineq',
                    'fun': self.constraints,
                    'args': (x0, t)}
        # con_eq = {'type': 'eq',
        #           'fun': self.con_eq,
        #           'args': (x0, t)}

        U = opt.minimize(self.cost_function, u0, args=(x0, t), method='SLSQP',
                         options={'maxiter': 200, 'disp': True}, constraints=[con_ineq])

        # U = opt.minimize(self.cost_function2, u0, args=(x0, t), method='SLSQP',
        #                  options={'maxiter': 200, 'disp': True})
        U = U.x
        return U

    def get_state_and_input(self, u0, x0):
        X, U = np.zeros((len(x0), len(self.t))), np.zeros((len(u0), len(self.t)))
        u0 = np.tile(u0, (self.N, 1))
        nx, nu = self.B.shape
        for i in range(len(self.t)):
            print('i = :', i)
            U[:, i], X[:, i] = u0[0:nu].transpose(), x0.transpose()
            u = self.optimise(u0, x0, i, )
            u0 = u
            u = u[0:nu].reshape(nu, -1)
            x0 = x0.reshape(len(x0), -1)
            x0 = self.A @ x0 + self.B @ u
        return X, U


if __name__ == '__main__':
    mpc = mpc_opt()
    pos = sp.zeros(3, len(mpc.t))
    x0, u0, = np.array([[0.0], [0.0], [0.0]]), np.array([[0.4], [0.2]])

    X, U = mpc.get_state_and_input(u0, x0)

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
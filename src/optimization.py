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
# This code is valid for Linear Systems

# TODO:
# 1. Code for adding control horizon
# 2.  Add jacobian in optimization to speed up


class mpc_opt():

    def __init__(self, A=None, B=None, C=None, Q=None, R=None, P=None,
                 xl=None, xh=None, ul=None, uh=None, N=4, x0=None, time=None, ref_traj=None):

        if not isinstance(A, (list, tuple, np.ndarray)):
            self.A = np.array([[-0.79, -0.3, -0.1], [0.5, 0.82, 1.23], [0.52, -0.3, -0.5]])
            self.B = np.array([[-2.04, -0.21], [-1.28, 2.75], [0.29, -1.41]])
            self.C = np.array([[0, 1, 0]])

            self.P = 4   # terminal state penalty
            self.Q = 5
            if isinstance(P, (list, tuple, np.ndarray)):
                self.P = self.C.transpose() @ self.P @ self.C
                self.P_dd = self.C.transpose() @ self.P
                self.Q = self.C.transpose() @ self.Q @ self.C
                self.Q_dd = self.C.transpose() @ self.Q
            else:
                self.P = self.P * self.C.transpose() @ self.C
                self.P_dd = self.P * self.C.transpose()
                self.Q = self.Q * self.C.transpose() @ self.C
                self.Q_dd = self.Q * self.C.transpose()
            self.R = np.eye(2)

            self.t = np.linspace(0, 1, 50)

            self.ul, self.uh, self.xl, self.xh = np.array([[-2.], [-3.]]), np.array([[2.], [3.]]), np.array([[-10.], [-9.], [-8.]]), np.array([[10.], [9.], [8.]])
            self.x0 = np.array([[0.0], [0.0], [0.0]])
            self.N = 3   # # Prediction horizon
            self.y_ref = np.array([[0], [2], [0]])

        else:
            # x_dot = A x + B u
            self.A = A  # linear system dynamics matrix
            self.B = B  # input matrix
            self.C = C  # output matrix

            if isinstance(P, (list, tuple, np.ndarray)):
                self.P = C.transpose() @ P @ C
                self.P_dd = C.transpose() @ P
                self.Q = C.transpose() @ Q @ C
                self.Q_dd = C.transpose() @ Q
            else:
                self.P = P * C.transpose() @ C
                self.P_dd = P * C.transpose()
                self.Q = Q * C.transpose() @ C
                self.Q_dd = Q * C.transpose()
            self.R = R

            # if self.C.shape[0] != self.P.shape[0]:
            #     raise ValueError(' DIMENSION MISMATCH: Both C and P should have same number of rows')

            self.t = time

            self.ul, self.uh, self.xl, self.xh = ul, uh, xl, xh
            self.x0 = x0
            self.N = N  # # Prediction horizon
            self.y_ref = ref_traj

        self.optCurve, self.costs = [], []
        self.omega = 0.5
        self.dyn = dynamics()
        self.kin = kinematics()

    def plant_model(self, x, u):
        x_dot = self.A * x + self.B * u
        return x_dot

    def transfer_matrices(self,):
        N = self.N
        nx, nu = self.B.shape
        Su = np.zeros((N * nx, N * nu))
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

    def plant_prediction(self, Sx, Su, x0, u):
        x = Sx @ x0 + Su @ u
        return x

    def penalties(self, ):
        N = self.N
        if N == 1:
            Qs = self.Q   # Q stacked as diagonal
            Qs_d = self.Q_dd
        else:
            Qs = np.kron(np.eye(N - 1), self.Q)   #  Q stacked as diagonal
            Qs_d = np.kron(np.eye(N - 1), self.Q_dd)
        return Qs, Qs_d, np.kron(np.eye(N), self.R), np.kron(np.eye(N), self.C)

    def block_diag(self, Q, P):
        N = self.N
        rQ, cQ = Q.shape
        if isinstance(P, (list, tuple, np.ndarray)):
            rP, cP = P.shape
        else:
            rP, cP = 1, 1
        t1 = np.zeros((rP+rQ, cQ+cP))
        if N != 1:
            t1[0:rQ, 0:cQ] = Q
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
        Qs, Qs_d, H, Cs = self.penalties()
        F = self.block_diag(Qs, self.P)
        G = self.block_diag(Qs_d, self.P_dd)

        W = 2 * (Su.transpose() @ F @ Su + H)
        K = 2 * Su.transpose() @ F @ Sx
        L = 2 * Su.transpose() @ G

        if self.y_ref.shape[1] > 1:
            y_ref = self.y_ref[:, t].reshape(len(self.y_ref[:, t]), -1)
        else:
            y_ref = self.y_ref
        y_ref_lifted = np.tile(y_ref, (N, 1))
        cost = 0.5 * u.transpose() @ W @ u + u.transpose() @ K @ x0 - u.transpose() @ L @ y_ref_lifted
        return cost

    # @staticmethod
    def ref_trajectory(self, i):
        return self.y_ref[:, i]

    def cost_gradient(self, u, *args):
        x0, t = args
        N = self.N
        u = u.reshape(len(u), -1)
        Sx, Su = self.transfer_matrices()
        Qs, Qs_d, Rs, Cs = self.penalties()
        Q_blk = self.block_diag(Qs, self.P)
        G = 2 * (Rs + Su.transpose() @ Q_blk @ Su)
        F = 2 * (Su.transpose() @ Q_blk @ Sx)
        K = 2 * Su.transpose() @ Q_blk
        # y_ref = self.ref_trajectory(t)
        y_ref_lifted = np.tile(self.y_ref, (N, 1))
        cost_gradient = np.vstack((0.5 * G @ u, F @ x0, -K @ y_ref_lifted, np.zeros((18, 1))))
        return np.squeeze(cost_gradient)

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

        # U = opt.minimize(self.cost_function, u0, args=(x0, t), method='SLSQP',
        #                  options={'maxiter': 200, 'disp': True}, jac=self.cost_gradient, constraints=con_ineq)

        U = opt.minimize(self.cost_function, u0, args=(x0, t), method='SLSQP',
                         options={'maxiter': 200, 'disp': True},)# constraints=con_ineq)
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

    cg = mpc.cost_gradient(np.tile(u0, (mpc.N, 1)), x0, 1)
    cons = mpc.constraints(np.tile(u0, (mpc.N, 1)), x0, 1)

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
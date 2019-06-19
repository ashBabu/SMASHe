import numpy as np
import sympy as sp
import scipy.optimize as opt
from mechanics import dynamics, kinematics
import matplotlib.pyplot as plt
import scipy.sparse as sparse


class mpc_opt():

    def __init__(self):
        self.optCurve, self.costs = [], []
        self.omega = 0.5
        # self.Q = 0.15*sp.eye(3)
        # self.R = 0.5*sp.eye(2)
        # self.A = sp.Matrix([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        # self.B = sp.Matrix([[0, 0], [0, 0], [1, 0], [0, 1]])
        self.Q = 5*np.diag((0.5, 1, 0))
        self.R = np.eye(2)
        # self.A = sp.Matrix([[-0.79, -0.3, -0.1],[0.5, 0.82, 1.23], [0.52, -0.3, -0.5]])
        # self.B = sp.Matrix([[-2.04, -0.21], [-1.28, 2.75], [0.29, -1.41]])

        self.A = np.array([[-0.79, -0.3, -0.1], [0.5, 0.82, 1.23], [0.52, -0.3, -0.5]])
        self.B = np.array([[-2.04, -0.21], [-1.28, 2.75], [0.29, -1.41]])
        self.C = 2*np.eye(3)

        # self.B = np.array([[-2.04, -0.21], [-1.28, 2.75], [0.29, -1.41]])
        # self.B = 3*np.eye(3, 2)
        # self.A = 2*np.eye(3)

        self.t = np.linspace(0, 1, 30)

        self.dyn = dynamics()
        self.kin = kinematics()

    def plant_model(self, x, u):
        x_dot = self.A * x + self.B * u
        return x_dot

    def transfer_matrices(self, ny):
        [nx, nB] = self.B.shape
        Su = np.zeros((ny*nx, ny*nB))
        Sx = self.C @ self.A
        I = self.A @ self.A
        for i in range(ny-1):
            Sx = np.concatenate((Sx, self.C @ I), axis=0)
            I = I @ self.A
        for i in range(ny):
            Su = Su + np.kron(np.eye(ny, k=-i), self.C @ self.B)
            B = self.A @ self.B
        return Sx, Su

    def penalties(self, ny):
        return np.kron(np.eye(ny), self.Q), np.kron(np.eye(ny), self.R), np.kron(np.eye(ny), self.C)

    def plant(self, Sx, Su, x0, u):
        x = Sx @ x0 + Su @ u
        return x

    def cost_function2(self, u, ny, x0):
        Sx, Su = self.transfer_matrices(ny)
        Qs, Rs, Cs = self.penalties(ny)
        y = Cs @ self.plant(Sx, Su, x0, u)
        y_ref = self.ref_trajectory(1)
        y_ref_lifted = np.tile(y_ref, (ny, 1))
        temp = y_ref_lifted - y
        J = temp.transpose() @ Qs @ temp + u.transpose() @ Rs @ u
        return J

    def end_effec_pose(self, q):
        pos, _ = self.kin.fwd_kin_numeric(self.kin.l, q)
        return pos

    @staticmethod
    def ref_trajectory(i):  # y = 3*sin(2*pi*omega*t)
        # y = 3 * np.sin(2*np.pi*self.omega*self.t[i])
        x_ref = np.array([[0], [1], [0]])
        return x_ref
        # return sp.Matrix(([[self.t[i]], [y], [0]]))

    def cost_function(self, U, *args):
        t = args
        nx, nu = self.B.shape
        x0 = U[0:nx]
        u = U[nx:nx+nu]
        u = u.reshape(len(u), -1)
        x0 = x0.reshape(len(x0), -1)
        x1 = self.A * x0 + self.B * u
        # q = [x1[0], x1[1]]
        # pos = self.end_effec_pose(q)
        traj_ref = self.ref_trajectory(t)
        pos_error = x1 - traj_ref
        cost = pos_error.transpose() * self.Q * pos_error + u.transpose() * self.R * u
        return cost

    def cost_gradient(self, U, *args):
        t = args
        nx, nu = self.A.shape[-1], self.B.shape[-1]
        x0 = U[0:nx]
        u = U[nx:nx + nu]
        u = u.reshape(len(u), -1)
        x0 = x0.reshape(len(x0), -1)
        x1 = self.A * x0 + self.B * u
        traj_ref = self.ref_trajectory(t)
        pos_error = x1 - traj_ref
        temp1 = self.Q * pos_error
        cost_gradient = temp1.col_join(self.R * u)
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

    def optimise(self, u0, t):
        umin = [-2., -3.]
        umax = [2., 3.]
        xmin = [-10., -9., -8.]
        xmax = [10., 9., 8.]
        bounds = ((xmin[0], xmax[0]), (xmin[1], xmax[1]), (xmin[2], xmax[2]), (umin[0], umax[0]), (umin[1], umax[1]))

        U = opt.minimize(self.cost_function2, u0, args=(t), method='SLSQP', bounds=bounds,
                         options={'maxiter': 200, 'disp': True})
        U = U.x
        return U


if __name__ == '__main__':
    mpc = mpc_opt()
    mpc.transfer_matrices(5)
    # P, H = mpc.imgpc_predmat(mpc.A, mpc.B, np.eye(3), 0, 5)
    pos = sp.zeros(3, len(mpc.t))
    x0, u0, = sp.Matrix([[0.1], [0.02], [0.05]]), sp.Matrix([[0.4], [0.2]])
    X, U = sp.zeros(len(x0), len(mpc.t)), sp.zeros(len(u0), len(mpc.t))
    # x0, u0, U, X = sp.Matrix([[0.1], [0.01], [0.01], [0.01]]), sp.Matrix([[0.1], [0.2]]), [], []
    U0 = sp.Matrix([x0, u0])
    nx, nu = mpc.A.shape[-1], mpc.B.shape[-1]
    for i in range(len(mpc.t)):
        print('i = :', i)
        result = mpc.optimise(U0, i)
        x0 = result[0:nx]
        u = result[nx:nx + nu]
        u = u.reshape(len(u), -1)
        x0 = x0.reshape(len(x0), -1)
        U[:, i], X[:, i] = u0, x0
        # x0 = mpc.A * x0 + mpc.B * u
        U0 = result

    plt.plot(X[0, :], '--r')
    # plt.plot(X[1, :], '--b')
    # plt.plot(X[2, :], '*r')
    plt.show()

    # y = 3 * np.sin(2*np.pi*mpc.omega*mpc.t)
    # plt.plot(mpc.t, y, '*b')
    # plt.hold(True)
    # plt.plot(pos[0, :], pos[1, :],  '-r')

print ('hi')
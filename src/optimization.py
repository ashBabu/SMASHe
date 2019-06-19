import numpy as np
import sympy as sp
import scipy.optimize as opt
from mechanics import dynamics, kinematics
import matplotlib.pyplot as plt


class mpc_opt():

    def __init__(self):
        self.optCurve, self.costs = [], []
        self.Q = 0.15*sp.eye(3)
        self.R = 0.5*sp.eye(2)
        self.omega = 0.5
        # self.A = sp.Matrix([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        # self.B = sp.Matrix([[0, 0], [0, 0], [1, 0], [0, 1]])
        self.A = sp.Matrix([[-0.79, -0.3, -0.1],[0.5, 0.82, 1.23], [0.52, -0.3, -0.5]])
        self.B = sp.Matrix([[-2.04, -0.21], [-1.28, 2.75], [0.29, -1.41]])

        self.t = np.linspace(0, 1, 10)

        self.dyn = dynamics()
        self.kin = kinematics()

    def plant_model(self, x, u):
        x_dot = self.A * x + self.B * u
        return x_dot

    def ref_trajectory(self, i):  # y = 3*sin(2*pi*omega*t)
        y = 3 * np.sin(2*np.pi*self.omega*self.t[i])
        return sp.Matrix(([[self.t[i]], [y], [0]]))

    def end_effec_pose(self, q):
        pos, _ = self.kin.fwd_kin_numeric(self.kin.l, q)
        return pos
    # reference trajectory  ## static!!!

    def cost_function(self, u, *args):
        x0, t = args
        u = u.reshape(len(u), -1)
        x1 = self.A * x0 + self.B * u
        q = [x1[0], x1[1]]
        pos = self.end_effec_pose(q)
        traj_ref = self.ref_trajectory(t)
        pos_error = pos - traj_ref
        cost = pos_error.transpose() * self.Q * pos_error + u.transpose() * self.R * u
        return cost

    def optimise(self, u0, x0, t):
        U = opt.minimize(self.cost_function, u0, args=(x0, t), method='SLSQP',
                                 options={'maxiter': 50, 'disp': True})
        U = U.x
        return U


if __name__ == '__main__':
    mpc = mpc_opt()
    pos = sp.zeros(3, len(mpc.t))
    x0, u0, U, X = sp.Matrix([[0.1], [0.01], [0.01], [0.01]]), sp.Matrix([[0.1], [0.2]]), [], []
    for i in range(len(mpc.t)):
        print('i = :', i)
        U.append(mpc.optimise(u0, x0, i))
        X.append(x0)
        q = [x0[0], x0[1]]
        pos[:, i] = mpc.end_effec_pose(q)
        print('end-eff pos = : ', mpc.end_effec_pose(q))
        x0 = mpc.A * x0 + mpc.B * sp.Matrix(U[-1])
        u0 = U[-1]
    y = 3 * np.sin(2*np.pi*mpc.omega*mpc.t)
    plt.plot(mpc.t, y, '*b')
    # plt.hold(True)
    plt.plot(pos[0, :], pos[1, :],  '-r')

print ('hi')
import numpy as np
import sympy as sp
import scipy.optimize as opt
from mechanics import dynamics, kinematics


class mpc_opt():

    def __init__(self):
        self.optCurve, self.costs = [], []
        self.Q = sp.eye(3)
        self.R = sp.eye(2)
        self.omega = 0.5
        self.A = sp.Matrix([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.B = sp.Matrix([[0, 0], [0, 0], [1, 0], [0, 1]])

        self.dyn = dynamics()
        self.kin = kinematics()

    # reference trajectory
    def ref_trajectory(self, t):  # y = 3*sin(2*pi*omega*t)
        y = 3 * np.sin(2*np.pi*self.omega*t)
        return sp.Matrix(([[t], [y], [0]]))  ######################### 0 added

    def cost_function(self, u, *args):
        x0, t = args
        u = u.reshape(len(u), -1)
        x_1 = self.A * x0 + self.B * u
        q = [x_1[0], x_1[1]]
        pos, _ = self.kin.fwd_kin_numeric(self.kin.l, q)
        traj_ref = self.ref_trajectory(t)
        temp1 = pos - traj_ref
        return temp1.transpose() * self.Q * temp1 + u.transpose() * self.R * u

    def cost_gradient(self, u, *args):
        x0, t = args
        u = u.reshape(len(u), -1)
        x_1 = self.A * x0 + self.B * u
        q = [x_1[0], x_1[1]]
        pos, _ = self.kin.fwd_kin_numeric(self.kin.l, q)
        traj_ref = self.ref_trajectory(t)
        temp1 = pos - traj_ref
        return self.Q * temp1 + self.R * u

    def optim_callback(self, xk):
        self.costs.append(self.cost_function(xk)[0, 0])
        self.optCurve.append(xk)
        print ('Iteration {}: {:2.4}\n').format(len(self.optCurve), self.costs[len(self.optCurve) - 1])

    def optimise(self, u0, x0, t):
        # U = opt.minimize(self.cost_function, u0, args=(x0, t), method='BFGS',
        #                          jac=self.cost_gradient, options={'maxiter': 50, 'disp': True},
        #                          callback=self.optim_callback)
        U = opt.minimize(self.cost_function, u0, args=(x0, t), method='SLSQP',
                                 options={'maxiter': 50, 'disp': True}, callback=self.optim_callback )
        U = U.x
        return U, self.costs, U.success


if __name__ == '__main__':
    mpc = mpc_opt()
    x = mpc.ref_trajectory(1)
    t = np.linspace(0, 10, 100)
    x0, u0, U = sp.Matrix([[0.1], [0.01], [0.01], [0.01]]), sp.Matrix([[0.1], [0.2]]), []
    for i in range(len(t)):
        U, _, _ = mpc.optimise(u0, x0, t[i])
        x0 = mpc.A * x0 + mpc.B * U
        U.append(U)
    print ('hi')
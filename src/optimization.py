import numpy as np
import sympy as sp
import time
import scipy.optimize as opt
from mechanics import dynamics, kinematics


class mpc_opt():

    def __init__(self):
        self.optCurve, self.costs = [], []
        self.Q = sp.eye(4)
        self.R = sp.eye(4)
        self.omega = 0.5

        self.dyn = dynamics()
        self.kin = kinematics()

    # reference trajectory
    def ref_trajectory(self, t):  # y = 3*sin(2*pi*omega*t)
        y = 3 * np.sin(2*np.pi*self.omega*t)
        return np.array([t, y])

    def cost_function(self, q, *args):
        u, t = args
        pos, _ = self.kin.fwd_kin_numeric(self.kin.l, q)
        traj_ref = self.ref_trajectory(t)
        temp1 = pos - traj_ref
        return temp1.transpose() * self.Q * temp1 + u.transpose() * self.R * u

    def cost_gradient(self, q, *args):
        u, t = args
        pos, _ = self.kin.fwd_kin_numeric(self.kin.l, q)
        traj_ref = self.ref_trajectory(t)
        temp1 = pos - traj_ref
        return self.Q * temp1 + self.R * u

    def optim_callback(self, xk):
        self.costs.append(self.cost_function(xk)[0, 0])
        self.optCurve.append(xk)
        print 'Iteration {}: {:2.4}\n'.format(len(self.optCurve), self.costs[len(self.optCurve) - 1])

    def optimise(self, x0, u, t):
        opti_traj = opt.minimize(self.cost_function, x0, args=(u, t), method='BFGS',
                                     jac=self.cost_gradient, options={'maxiter': 150, 'disp': True},
                                     callback=self.optim_callback)
        opti_trajec = opti_traj.x
        return opti_trajec, self.costs, opti_traj.success


if __name__ == '__main__':
    mpc = mpc_opt()
    x = mpc.ref_trajectory()
    print 'hi'
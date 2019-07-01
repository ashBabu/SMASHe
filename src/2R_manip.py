import numpy as np
import matplotlib.pyplot as plt
from optimization import mpc_opt
# from test2 import mpc_opt
from mechanics import dynamics, kinematics


class Rmanip_mpc():
    def __init__(self):
        self.q = 0
        self.kin = kinematics()
        self.dyn = dynamics()

        # Matrices after feedback linearization as described in Ghosal. # x_dot = A x + B u
        self.A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
        self.C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # Penalties
        self.Q = 100 * np.diag((0.5, 1))  # state penalty
        self.P = 1 * np.diag((1.5, 1.5))  # terminal state penalty
        self.R = 5*np.eye(2)  # input penalty

        self.R = np.eye(2)
        self.t = np.linspace(0, 10, 50)  # sampling time
        self.ul, self.uh, self.xl, self.xh = np.array([[-10.], [-30.]]), np.array([[20.], [30.]]), np.array(
            [[-10.], [-9.], [-8.], [-8.]]), np.array([[10.], [9.], [8.], [8.]])  # input and state lower and upper limits
        self.N = 3  # Prediction horizon (control horizon is assumed to be the same)
        self.x0, self.u0, = np.array([[0.0], [0.0], [0.0], [0.0]]), np.array([[0.4], [0.2]])  # initial state and input guess

        self.mpc = mpc_opt(Q=self.Q, P=self.P, R=self.R, A=self.A, B=self.B, C=self.C, time=self.t, ul=self.ul,
                           uh=self.uh, xl=self.xh, N=self.N, ref_traj=self.ref_traj_joint_space())

    def end_effec_pose(self, q):
        pos, _ = self.kin.fwd_kin_numeric(self.kin.l, q)
        return pos

    def ref_traj(self,):   #  track a circle or radius 0.5 and centre at 1.5
        r = 0.5
        x = 1.0 + r * np.cos(self.t)
        y = 0.0 + r * np.sin(self.t)
        return np.array([x, y])

    def ref_traj_joint_space(self, ):
        X = self.ref_traj()
        q1, q2 = np.zeros(X.shape[1]), np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            q1[i], q2[i] = self.kin.inv_kin(X[:, i])
        return np.vstack((q1, q2))

    def plotter(self, X, Y):
        fig = plt.figure()
        ref_traj = self.ref_traj()
        plt.plot(ref_traj[0, :], ref_traj[1, :])
        ax = fig.add_subplot(111)
        plt.show(block=False)
        x, y, z = 0, 0, 0
        for i in range(X.shape[1]):
            ax.plot([x, X[0, i]], [y, X[1, i]])
            ax.plot([X[0, i], Y[0, i]], [X[1, i], Y[1, i]])
            plt.xlabel('X')
            plt.ylabel('Y')
            scale = 2
            ax.set_xlim(-1 * scale, 1 * scale)
            ax.set_ylim(-1 * scale, 1 * scale)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.pause(0.1)


if __name__ == '__main__':

    two_R = Rmanip_mpc()
    q = two_R.ref_traj_joint_space()
    X, U = two_R.mpc.get_state_and_input(two_R.u0, two_R.x0)
    q = X[0:2, :]
    pos, x, y = np.zeros((3, q.shape[1])), np.zeros((2, q.shape[1])), np.zeros((q.shape[1]))
    for i in range(q.shape[1]):
        temp = np.array(two_R.end_effec_pose(q[:, i])).astype(np.float64)
        pos[:, i] = temp.reshape(-1, len(temp))
        a = np.array(two_R.kin.l[0] * np.cos(q[0, i])).astype(np.float64)
        b = np.array(two_R.kin.l[0] * np.sin(q[0, i])).astype(np.float64)
        x[:, i] = a, b #np.vstack((a, b))
    ref_traj = two_R.ref_traj()
    plt.figure(3)
    # plt.plot(ref_traj.transpose())
    plt.plot(ref_traj[0, :], ref_traj[1, :])
    plt.plot(pos[0, :], pos[1, :])
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    two_R.plotter(x, pos)
    print('hi')
    # plt.figure(1)
    # plt.plot(X.transpose())
    # plt.xlabel('time')
    # plt.title('states')
    #
    # plt.figure(2)
    # plt.plot(U.transpose())
    # plt.title('Input')
    # plt.xlabel('time')
    #
    # # plt.show()
    #
    # print('hi')
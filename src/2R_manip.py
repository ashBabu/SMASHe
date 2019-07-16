import numpy as np
import matplotlib.pyplot as plt
from mpc_optimizer import mpc_opt
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
        self.P = 10 * np.diag((1.5, 1.5))  # terminal state penalty
        self.R = 10 * np.diag((0.5, 1))  # state penalty
        # self.R = 5*np.eye(2)  # input penalty

        self.R = np.eye(2)
        self.t = np.linspace(0, 10, 50)  # sampling time
        self.ul, self.uh, self.xl, self.xh = np.array([[-10.], [-30.]]), np.array([[20.], [30.]]), np.array(
            [[-10.], [-9.], [-8.], [-8.]]), np.array([[10.], [9.], [8.], [8.]])  # input and state lower and upper limits
        self.N = 5  # Prediction horizon (control horizon is assumed to be the same)
        self.x0, self.u0, = np.array([[0.1], [0.2], [0.4], [0.1]]), np.array([[1.4], [2.3]])  # initial state and input guess

    def end_effec_pose(self, q):
        pos, _ = self.kin.fwd_kin_numeric(self.kin.ln, q)
        return pos

    def ref_traj(self,):   # track a circle or radius 'r' and centre at (x, y)
        r = 0.5
        x = -1.0 + r * np.cos(self.t)
        y = 1.0 + r * np.sin(self.t)
        return np.array([x, y])

    def ref_traj2(self):  # To track a figure 8 whose parametric equation is (a + cos(t), b + cos(t)*sin(t))
        a, b = -0.5, 1.0
        x = a + np.cos(self.t)
        y = b + np.sin(self.t) * np.cos(self.t)
        return np.array([x, y])

    def ref_traj_joint_space(self, ref):
        X = ref
        q1, q2 = np.zeros(X.shape[1]), np.zeros(X.shape[1])
        q0 = np.array([0, 0])
        for i in range(X.shape[1]):
            # q1[i], q2[i] = self.kin.inv_kin2(q0, X[:, i])
            q1[i], q2[i] = self.kin.inv_kin(X[:, i])
            q0 = np.array([q1[i], q2[i]])
        return np.vstack((q1, q2))

    def plotter(self, X, Y, ref):
        fig = plt.figure()
        ref_traj = ref
        plt.plot(ref_traj[0, :], ref_traj[1, :])
        ax = fig.add_subplot(111)
        # plt.show(block=False)
        x, y, z = 0, 0, 0
        for i in range(X.shape[1]):
            ax.plot([x, X[0, i]], [y, X[1, i]])
            ax.plot([X[0, i], Y[0, i]], [X[1, i], Y[1, i]])
            plt.xlabel('X')
            plt.ylabel('Y')
            scale = 2
            ax.set_xlim(-1 * scale, 1 * scale)
            ax.set_ylim(-1 * scale, 1 * scale)
            # fig.canvas.draw()
            # fig.canvas.flush_events()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.pause(0.1)
        # plt.show(block=True)
            # plt.clf()
            # plt.hold(False)


if __name__ == '__main__':

    two_R = Rmanip_mpc()
    ref_circle = two_R.ref_traj()
    ref_eight = two_R.ref_traj2()
    ref_traj = two_R.ref_traj2()
    two_R.mpc = mpc_opt(Q=two_R.Q, P=two_R.P, R=two_R.R, A=two_R.A, B=two_R.B, C=two_R.C, time=two_R.t, ul=two_R.ul,
                       uh=two_R.uh, xl=two_R.xh, N=two_R.N, ref_traj=two_R.ref_traj_joint_space(ref_traj))

    q = two_R.ref_traj_joint_space(ref_traj)
    X, U = two_R.mpc.get_state_and_input(two_R.u0, two_R.x0)
    # q = X[0:2, :]
    pos, x, y = np.zeros((3, q.shape[1])), np.zeros((2, q.shape[1])), np.zeros((q.shape[1]))
    for i in range(q.shape[1]):  # to get the end-point of 1st link
        temp = np.array(two_R.end_effec_pose(q[:, i])).astype(np.float64)
        pos[:, i] = temp.reshape(-1, len(temp))
        a = np.array(two_R.kin.ln[0] * np.cos(q[0, i])).astype(np.float64)
        b = np.array(two_R.kin.ln[0] * np.sin(q[0, i])).astype(np.float64)
        x[:, i] = a, b #np.vstack((a, b))

    lp, q_dot = two_R.kin.ln, X[2:4, :]
    inp = np.zeros((2, q.shape[1]))
    for i in range(q.shape[1]):
        M, C, G = two_R.dyn.dyn_para_numeric(lp, q[:, i], q_dot[:, i])
        M, C, G = np.array(M).astype(np.float64), np.array(C).astype(np.float64), np.array(G).astype(np.float64)
        temp = M @ U[:, i].reshape((2, 1)) + C + G
        inp[:, i] = temp[0, 0], temp[1, 0]

    two_R.plotter(x, pos, ref_traj)    # Animation
    plt.figure()
    plt.plot(U.transpose())
    plt.title('Linear_system_input')
    plt.xlabel('time')
    #
    plt.figure()
    plt.plot(ref_traj[0, :], ref_traj[1, :], '--r')
    plt.plot(pos[0, :], pos[1, :], '*b')
    plt.gca().set_aspect('equal', adjustable='box')

    # print('hi')

    plt.figure()
    plt.plot(inp.transpose())
    plt.xlabel('time')
    plt.title('actual_inputs')
    plt.show(block=True)
    #
    #

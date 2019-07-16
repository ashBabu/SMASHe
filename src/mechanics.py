import numpy as np
from sympy import *
from sympy.physics.mechanics import *
from sympy.tensor.array import Array
from decimal import getcontext
# import cvxpy as cp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.optimize as opt
from sympy.physics.vector import ReferenceFrame, Vector
from sympy.physics.vector import time_derivative


# Set the precision.
getcontext().prec = 3
nDoF = 6


class kinematics():

    def __init__(self):
        self.q1, self.q2 = dynamicsymbols('q1 q2')
        self.q1d, self.q2d = dynamicsymbols('q1 q2', 1)
        self.q = [self.q1, self.q2]
        self.qd = [self.q1d, self.q2d]
        self.l1, self.l2 = symbols('l_1 l_2', positive=True)
        self.l = [self.l1, self.l2]
        self.ln = [1.5, 1.0]  ############################

        # COM vectors
        self.r1, self.r2 = symbols('r1 r2')
        self.r11 = zeros(3, 1)
        # self.r11[0] = self.r1
        self.r11[0] = 1  ################################
        self.r22 = zeros(3, 1)
        # self.r22[0] = self.r2
        self.r22[0] = 1  #################################
        self.r = zeros(3, 2)
        self.r[:, 0] = self.r11
        self.r[:, 1] = self.r22

        self.a = Array([0, self.l[0], self.l[1]])
        self.d = Array([0.0, 0.0, 0.0])
        self.alpha = Array([0.0, 0.0, 0.0])
        self.T_eff = eye(4)
        self.T_eff[0, 3] = self.l[-1]

        self.q_i = Symbol("q_i")
        self.alpha_i = Symbol("alpha_i")
        self.a_i = Symbol("a_i")
        self.d_i = Symbol("d_i")

    # def initializing(self, nDoF):
    #     # q = dynamicsymbols('q:{0}'.format(nDoF))
    #     # r = dynamicsymbols('r:{0}'.format(nDoF))
    #     # for i in range(1, nDoF+1):
    #     q = dynamicsymbols(['q%d' % x for x in range(1, nDoF+1)])
    #     qd = dynamicsymbols(["qd%d" % x for x in range(1, nDoF+1)], 1)
    #     l = symbols(["l%d" % x for x in range(1, nDoF+1)])
    #     m = symbols(["m%d" % x for x in range(1, nDoF+1)])
    #     I = symbols(["I%d" % x for x in range(1, nDoF+1)])
    #     tau = symbols(["tau%d" % x for x in range(1, nDoF+1)])
    #     g = symbols('g', positive=True)
    #     grav = transpose(Matrix([[0, g, 0]]))
    #     # r = dynamicsymbols(["r%d" % x for x in range(1, nDoF+1)])
    #     # qd = dynamicsymbols('r:{0}'.format(nDoF))
    #     # mass and length
    #     m = symbols('m:{0}'.format(nDoF))
    #     l = symbols('l:{0}'.format(nDoF))
    #     # gravity and time symbols
    #     g, t = symbols('g, t')
    #     return q, m, l, g

    # def geometric_jacobian(self, T_joint,):  # Method 2: for finding Jacobian
    #     num_joints = len(self.a) - 1
    #     Jac = zeros(num_joints, num_joints)  # initilizing jacobian
    #     for i in range(len(T_joint) - 1):
    #         pos_vec_cm = T_joint[i][0:3, 3] + (T_joint[i+1][0:3, 3] - T_joint[i][0:3, 3])/2
    #         rot_axis = T_joint[i][0:3, 2]
    #         Jac[0:3, i] = rot_axis.cross(pos_vec_cm)
    #         Jac[3:6, i] = rot_axis
    #     return Jac

    # def robot_cms(self, q):
    #     T_joint, _ = self.fwd_kin_symbolic(q)
    #     pos_vec_com = Matrix.zeros(3, len(q) - 1)
    #     # pos_vec_com = []
    #     for i in range(len(T_joint) - 1):
    #         temp1 = T_joint[i][0:3, 3] + (T_joint[i+1][0:3, 3] - T_joint[i][0:3, 3])/2
    #         pos_vec_com[:, i] = temp1
    #     return pos_vec_com

    def fwd_kin_symbolic(self, q):
        T = Matrix([[cos(self.q_i), -sin(self.q_i), 0, self.a_i],
                    [sin(self.q_i) * cos(self.alpha_i), cos(self.q_i) * cos(self.alpha_i), -sin(self.alpha_i), -sin(self.alpha_i) * self.d_i],
                    [sin(self.q_i) * sin(self.alpha_i), cos(self.q_i) * sin(self.alpha_i), cos(self.alpha_i), cos(self.alpha_i) * self.d_i],
                    [0, 0, 0, 1]])
        T_joint, T_i_i1 = [], []  # T_i_i1 is the 4x4 transformation matrix relating i+1 frame to i
        t = eye(4)
        for i in range(len(q)):
            temp = T.subs(self.alpha_i, self.alpha[i]).subs(self.a_i, self.a[i]).subs(self.d_i, self.d[i]).subs(self.q_i, self.q[i])
            t = t*temp
            T_joint.append(t)
            T_i_i1.append(temp)
        return T_joint, T_i_i1

    def fwd_kin_numeric(self, lp, qp):  # provide the values of link lengths and joint angles to get the end-eff pose
        T_joint, _ = self.fwd_kin_symbolic(self.q)
        T_0_eff = T_joint[-1] * self.T_eff
        # qp, lp = [0, np.pi/2], [1, 1]
        for i in range(len(self.q)):
            T_0_eff = T_0_eff.subs(self.q[i], qp[i]).subs(self.l[i], lp[i])
        Rot_0_eff = T_0_eff[0:3, 0:3]
        pos_0_eff = T_0_eff[0:3, 3]
        return pos_0_eff, Rot_0_eff

    def inv_kin(self, X):
        a = X[0]**2 + X[1]**2 - self.ln[0]**2 - self.ln[1]**2
        b = 2 * self.ln[0] * self.ln[1]
        q2 = np.arccos(a/b)
        c = np.arctan2(X[1], X[0])
        q1 = c - np.arctan2(self.ln[1] * np.sin(q2), (self.ln[0] + self.ln[1]*np.cos(q2)))
        return q1, q2

    def inv_kin_optfun(self, q):
        # a = np.array(two_R.end_effec_pose(q[:, i])).astype(np.float64)
        pos_0_eff, _ = self.fwd_kin_numeric(self.l, q)
        pos_0_eff = np.array(pos_0_eff[0:2]).astype(np.float64)
        k = pos_0_eff.reshape(-1) - self.T_desired
        # k = np.reshape(k, (16, 1))
        k = k.transpose() @ k
        return k

    def inv_kin2(self, q_current, T_desired):  # Implements Min. (F(q) - T_desired) = 0
        x0 = q_current
        self.T_desired = T_desired
        final_theta = opt.minimize(self.inv_kin_optfun, x0,
                                   method='BFGS', )  # jac=self.geometric_jacobian(T_joint, T_current))
        # print 'res \n', final_theta
        # final_theta = np.insert(final_theta.x, self.numJoints, 0)  # Adding 0 at the end for the fixed joint

        return final_theta.x

    def velocities(self, q):
        omega = Matrix.zeros(3, len(q)+1)
        joint_velocity = Matrix.zeros(3, len(q)+1)
        cm_vel = Matrix.zeros(3, len(q))
        _, t_i_i1 = self.fwd_kin_symbolic(q)  # T_i_i1 is the 4x4 transformation matrix of i+1 frame wrt to i
        for i in range(len(q)):
            R = t_i_i1[i][0:3, 0:3].transpose()
            omega[:, i+1] = R * omega[:, i] + Matrix([[0], [0], [self.qd[i]]])
            joint_velocity[:, i+1] = R * (joint_velocity[:, i] + omega[:, i].cross(t_i_i1[i][0:3, 3]))
        omega, joint_velocity = omega[:, 1:], joint_velocity[:, 1:]
        for i in range(len(q)):
            # cm_vel[:, i] = joint_velocity[:, i] + omega[:, i].cross(t_i_i1[i][0:3, 3]/2)
            cm_vel[:, i] = joint_velocity[:, i] + omega[:, i].cross(self.r[:, i])
        return omega, cm_vel, joint_velocity


class dynamics():

    def __init__(self):
        self.tau_1, self.tau_2, self.I1_zz, self.I2_zz, self.m1, self.m2 = symbols('tau_1 tau_2 I1_zz, I2_zz, m1, m2')
        self.g = symbols('g', positive=True)
        # self.m = [self.m1, self.m2]
        self.m = [3, 1] #############################
        self.grav = transpose(Matrix([[0, self.g, 0]]))

        # Inertia tensor wrt centre of mass of each link
        self.I1 = zeros(3, 3)
        # self.I1[2, 2] = self.I1_zz
        self.I1[2, 2] = 2   ###########################
        self.I2 = zeros(3, 3)
        # self.I2[2, 2] = self.I2_zz
        self.I2[2, 2] = 1.5   ##########################
        self.I = [self.I1, self.I2]

        self.kin = kinematics()
        self.M, self.C, self.G = self.get_dyn_para(self.kin.q, self.kin.qd)


    def kinetic_energy(self, q):
        w, cm_vel, _ = self.kin.velocities(q)
        K = 0
        for i in range(len(q)):
            K += 0.5*self.m[i]*cm_vel[:, i].dot(cm_vel[:, i]) + 0.5*w[:, i].dot(self.I[i]*w[:, i])
        return K

    def potential_energy(self, q):
        T_joint, _ = self.kin.fwd_kin_symbolic(q)  # T_joint is the 4x4 transformation matrix relating i_th frame  wrt to 0
        P = 0
        for i in range(len(q)):
            r_0_cm = T_joint[i][0:3, 0:3]*self.kin.r[:, i] + T_joint[i][0:3, 3]
            P += self.m[i]*self.grav.dot(r_0_cm)
        return P

    def get_dyn_para(self, q, qd):
        K = self.kinetic_energy(q)
        P = self.potential_energy(q)
        L = K - P  # Lagrangian
        M = transpose(Matrix([[K]]).jacobian(qd)).jacobian(qd).applyfunc(trigsimp)  # Mass matrix
        C = transpose(Matrix([[K]]).jacobian(qd)).jacobian(q) * Matrix(qd) - transpose(Matrix([[K]]).jacobian(q))  # Coriolis vector
        C = C.applyfunc(trigsimp)
        G = transpose(Matrix([[P]]).jacobian(q)).applyfunc(trigsimp)  # Gravity vector
        # LM = LagrangesMethod(L, q)
        # LM.form_lagranges_equations()
        # print LM.mass_matrix.applyfunc(trigsimp)
        # Matrix([P]).applyfunc(trigsimp)
        return M, C, G

    def dyn_para_numeric(self, lp, qp, q_dot):
        M, C, G = self.M, self.C, self.G
        for i in range(len(qp)):
            M = msubs(M, {self.kin.q[i]: qp[i], self.kin.l[i]: lp[i]})
            C = msubs(C, {self.kin.q[i]: qp[i], self.kin.l[i]: lp[i], self.kin.q[i].diff(): q_dot[i]})
            G = msubs(G, {self.kin.q[i]: qp[i], self.kin.l[i]: lp[i], self.g: 9.81})
        return M, C, G

    def round2zero(self, m, e):
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if (isinstance(m[i,j], Float) and m[i, j] < e):
                    m[i, j] = 0


if __name__ == '__main__':

    kin = kinematics()
    dyn = dynamics()
    lp, qp, q_dot = [1, 1], [0, np.pi/2], [0.1, 0.2]
    # M, C, G = dyn.get_dyn_para(kin.q, kin.qd)  # Symbolic dynamic parameters
    M, C, G = dyn.dyn_para_numeric(lp, qp, q_dot)  # Numeric values dynamic parameters
    print ('hi')





# calculate_jacobian(robot_cms(q), joint_index, q)

# print geometric_jacobian(T_joint,)
# link_1 = {'m': m1, 'com': [l1/2, 0, 0], 'd' : {'ixx': 0, 'ixy': 0.0, 'ixz': 0.0, 'iyy': 0.0, 'iyz': 0.0, 'izz': m1*l1**2/12}}
# link_2 = {'m': m2, 'com': [l2/2, 0, 0], 'd' : {'ixx': 0, 'ixy': 0.0, 'ixz': 0.0, 'iyy': 0.0, 'iyz': 0.0, 'izz': m2*l2**2/12}}

# B = ReferenceFrame('Base')  # Base (0) reference frame
# # J1 = CoordSys3D('Joint1', location=T01[0, 3]*B.i  + T01[1, 3]*B.j  + T01[2, 3]*B.k,  rotation_matrix=T01[0:3, 0:3].transpose(), parent=B)
# # J2 = CoordSys3D('Joint2', location=T12[0, 3]*J1.i + T12[1, 3]*J1.j + T12[2, 3]*J1.k, rotation_matrix=T12[0:3, 0:3].transpose(), parent=J1)
# # J3 = CoordSys3D('Joint3', location=T23[0, 3]*J2.i + T23[1, 3]*J2.j + T23[2, 3]*J2.k, rotation_matrix=T23[0:3, 0:3].transpose(), parent=J2)
# J1, J2, J3 = ReferenceFrame('Joint1'), ReferenceFrame('Joint2'), ReferenceFrame('Joint3')
# J1.orient(B,  'DCM', T01[0:3, 0:3].transpose())  # check: J1.dcm(B)
# J2.orient(J1, 'DCM', T12[0:3, 0:3].transpose())
# J3.orient(J2, 'DCM', T23[0:3, 0:3].transpose())
#
# J1.set_ang_vel(B, q1d * B.z)   # check: J1.ang_vel_in(B)
# J2.set_ang_vel(J1, q2d * J1.z)
# J3.set_ang_vel(J2, 0)
#
#
# O, O1, O2, O3 = Point('O'), Point('O1'), Point('O2'), Point('O3')
# O1.set_pos(O, 0)          #  O.locatenew('O1', 0)   : check: O1.pos_from(O)
# O2.set_pos(O1, l1*J1.x)   #  O1.locatenew('O2', l1*J1.x)
# O3.set_pos(O2, l2*J2.x)   #  O2.locatenew('O3', l2*J2.x)
#
# O.set_vel(B, 0)
# # O1.v2pt_theory(O, B, J1)
# # O2.v2pt_theory(O1, B, J2)
# # O3.v2pt_theory(O2, B, J3)
#
#
# CM_1, CM_2 = Point('CM_1'), Point('CM_2')
# CM_1.set_pos(O1, l1/2 * J1.x)
# V_CM1 = time_derivative(CM_1.pos_from(O).express(B), B)
# CM_1.set_vel(B, V_CM1)
# # CM_1.v2pt_theory(O1, B, J1)
#
# CM_2.set_pos(O2, l2/2 * J2.x)
# V_CM2 = time_derivative(CM_2.pos_from(O).express(B), B)
# CM_2.set_vel(B, V_CM2)
# # CM_2.v2pt_theory(O2, B, J2)
#
# I_l1 = inertia(J1, 0, 0, 0, 0, 0, m1*l1**2/12)
# I_l2 = inertia(J2, 0, 0, 0, 0, 0, m2*l2**2/12)
#
#
# link_1 = RigidBody('link_1', CM_1, J1, m1, (I_l1, CM_1))
# link_2 = RigidBody('link_2', CM_2, J2, m2, (I_l2, CM_2))
#
# # Potential energy
# P1 = -m1 * g * B.z
# l1_CM = CM_1.pos_from(O1).express(B)
# P2 = -m2 * g * B.z
# l2_CM = O2.pos_from(O1).express(B).simplify() + CM_2.pos_from(O2).express(B).simplify()
# PE = l1_CM.dot(P1) + l2_CM.dot(P2)
#
# v_l1_CM = time_derivative(l1_CM, B)
# v_l2_CM = time_derivative(l2_CM, B)
#
# KE = kinetic_energy(B, link_1, link_2)
# # KE1 = 0.5*m1*v_l1_CM.dot(v_l1_CM) + 0.5*m2*v_l2_CM.dot(v_l2_CM)
#
# FL = [(J1, tau_1 * B.z), (J2, tau_2 * B.z)]
#
# L = KE - PE
# L = L.simplify()
# LM = LagrangesMethod(L, [q1, q2], frame=B, forcelist=FL)
#
#
# L_eq = LM.form_lagranges_equations()
# print L_eq

#
# (link_2.potential_energy, link_1.potential_energy)
# OO1 = Vector(0)
# O1O2 = l1*J1.x
# O2O3 = l2*J2.x
#
# R_O1 = OO1
# V_O1 = time_derivative(R_O1, B)
# A_O1 = time_derivative(V_O1, B)
#
# R_O2 = R_O1 -T02[0:3, 0:3].transpose()*O1O2.express(B).to_matrix(B)
# V_O2 = time_derivative(R_O2, B)
# A_O2 = time_derivative(V_O2, B)
#
# R_O3 = R_O2 + -T03[0:3, 0:3].transpose()*O2O3.express(B).to_matrix(B)
# V_O3 = time_derivative(R_O3, B)
# A_O3 = time_derivative(V_O3, B)
#
# R_O1 = (R_O1.to_matrix(B)).applyfunc(trigsimp)
# V_O1 = (V_O1.to_matrix(B)).applyfunc(trigsimp)
# A_O1 = (A_O1.to_matrix(B)).applyfunc(trigsimp)
# R_O2 = (R_O2.to_matrix(B)).applyfunc(trigsimp)
# V_O2 = (V_O2.to_matrix(B)).applyfunc(trigsimp)
# A_O2 = (A_O2.to_matrix(B)).applyfunc(trigsimp)
# R_O3 = (R_O3.to_matrix(B)).applyfunc(trigsimp)
# V_O3 = (V_O3.to_matrix(B)).applyfunc(trigsimp)
# A_O3 = (A_O3.to_matrix(B)).applyfunc(trigsimp)
# pprint(R_O3)
# pprint(V_O3)
# pprint(A_O3)


# I1 = inertia(J1, 0.0, 0.0, m1*l1**2/3)
# I2 = inertia(J2, 0.0, 0.0, m2*l2**2/3)
#


# print J3.rotation_matrix(B).transpose()
# print express(J3.position_wrt(B), B)


# T_current, T_joint_current, T_j_prevC = fwd_kin(q)

# print T_current





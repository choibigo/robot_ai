from typing import Union

import matplotlib.pyplot as plt
import numpy as np

pinv_rcond = 1.4e-08


class VMP:
    def __init__(self, dim, kernel_num=30, kernel_std=0.1, elementary_type='linear', use_out_of_range_kernel=True):
        self.kernel_num = kernel_num
        if use_out_of_range_kernel:
            self.centers = np.linspace(1.2, -0.2, kernel_num)  # (K, )
        else:
            self.centers = np.linspace(1, 0, kernel_num)  # (K, )

        self.kernel_variance = kernel_std ** 2
        self.var_reci = - 0.5 / self.kernel_variance
        self.elementary_type = elementary_type
        self.lamb = 0.01
        self.dim = dim
        self.n_samples = 100
        self.kernel_weights = np.zeros(shape=(kernel_num, self.dim))

        self.h_params =None
        self.y0 = None
        self.g = None

    def __psi__(self, can_value: Union[float, np.ndarray]):
        """
        compute the contribution of each kernel given a canonical value
        """
        return np.exp(np.square(can_value - self.centers) * self.var_reci)

    def __Psi__(self, can_values: np.ndarray):
        """
        compute the contributions of each kernel at each time step as a (T, K) matrix, where
        can_value is a (T, ) array, the sampled canonical values, where T is the total number of time steps.
        """
        return self.__psi__(can_values[:, None])

    def h(self, x):
        if self.elementary_type == 'linear':
            return np.matmul(self.h_params, np.matrix([[1], [x]]))
        else:
            return np.matmul(self.h_params, np.matrix(
                [[1], [x], [np.power(x, 2)], [np.power(x, 3)], [np.power(x, 4)], [np.power(x, 5)]]))

    def linear_traj(self, can_values: np.ndarray):
        """
        compute the linear trajectory (T, dim) given canonical values (T, )
        """
        if self.elementary_type == 'linear':
            can_values_aug = np.stack([np.ones(can_values.shape[0]), can_values])
        else:
            can_values_aug = np.stack([np.ones(can_values.shape[0]), can_values, np.power(can_values, 2),
                                       np.power(can_values, 3), np.power(can_values, 4), np.power(can_values, 5)])
        return np.einsum("ij,ik->kj", self.h_params, can_values_aug)

    def train(self, trajectories):
        """
        Assume trajectories are regularly sampled time-sequences.
        """
        if len(trajectories.shape) == 2:
            trajectories = np.expand_dims(trajectories, 0)

        n_demo, self.n_samples, self.dim = trajectories.shape
        # self.dim -= 1

        can_value_array = self.can_sys(1, 0, self.n_samples)
        Psi = self.__Psi__(can_value_array)  # (T, K)

        if self.elementary_type == 'linear':
            y0 = trajectories[:, 0, :].mean(axis=0)
            g = trajectories[:, -1, :].mean(axis=0)
            self.h_params = np.stack([g, y0-g])

        else:
            # min_jerk
            y0 = trajectories[:, 0:3, 1:].mean(axis=0)
            g = trajectories[:, -2:, 1:].mean(axis=0)
            dy0 = (y0[1, 2:] - y0[0, 2:]) / (y0[1, 1] - y0[0, 1])
            dy1 = (y0[2, 2:] - y0[1, 2:]) / (y0[2, 1] - y0[1, 1])
            ddy0 = (dy1 - dy0) / (y0[1, 1] - y0[0, 1])
            dg0 = (g[1, 2:] - g[0, 2:]) / (g[1, 1] - g[0, 1])
            dg1 = (g[2, 2:] - g[1, 2:]) / (g[2, 1] - g[1, 1])
            ddg = (dg1 - dg0) / (g[1, 1] - g[0, 1])

            b = np.stack([y0[0, :], dy0, ddy0, g[-1, :], dg1, ddg])
            A = np.array([[1, 1, 1, 1, 1, 1], [0, 1, 2, 3, 4, 5], [0, 0, 2, 6, 12, 20], [1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0]])
            self.h_params = np.linalg.solve(A, b)

        self.y0 = y0
        self.g = g
        linear_traj = self.linear_traj(can_value_array)  # (T, dim)
        shape_traj = trajectories[..., :] - np.expand_dims(linear_traj, 0)  # (N, T, dim) - (1, T, dim)

        pseudo_inv = np.linalg.pinv(Psi.T.dot(Psi), pinv_rcond)  # (K, K)
        self.kernel_weights = np.einsum("ij,njd->nid", pseudo_inv.dot(Psi.T), shape_traj).mean(axis=0)

    def save_weights_to_file(self, stylename):
        filename = f"{stylename}_weights"
        np.savetxt(filename, self.kernel_weights, delimiter=',')

    def load_weights_from_file(self, stylename):
        filename = f'/workspace/data/weight/{stylename}_weights'
        self.kernel_weights = np.loadtxt(filename, delimiter=',')

    def get_weights(self):
        return self.kernel_weights

    def get_flatten_weights(self):
        return self.kernel_weights.flatten('F')

    def set_weights(self, ws: np.ndarray):
        """
        set weights to VMP

        Args:
            ws: (kernel_num, dim)
        """
        if np.shape(ws)[-1] == self.dim * self.kernel_num:
            self.kernel_weights = np.reshape(ws, (self.kernel_num, self.dim), 'F')
        elif np.shape(ws)[0] == self.kernel_num and np.shape(ws)[-1] == self.dim:
            self.kernel_weights = ws
        else:
            raise Exception(f"The weights have wrong shape. "
                            f"It should have {self.kernel_num} rows (for kernel number) "
                            f"and {self.dim} columns (for dimensions), but given is {ws.shape}.")

    def get_position(self, t):
        x = 1 - t
        return np.matmul(self.__psi__(x), self.kernel_weights)

    def set_start(self, y0):
        self.y0 = y0
        self.h_params = np.stack([self.g, self.y0 - self.g])

    def set_goal(self, g):
        self.g = g
        self.h_params = np.stack([self.g, self.y0 - self.g])

    def set_start_goal(self, y0, g):
        self.y0 = y0
        self.g = g
        self.h_params = np.stack([self.g, self.y0 - self.g])

    # def set_start_goal(self, y0, g, dy0=None, dg=None, ddy0=None, ddg=None):
    #     self.y0 = y0
    #     self.g = g
    #     self.q0 = y0
    #     self.q1 = g
    #
    #     self.goal = g
    #     self.start = y0
    #
    #
    #     if self.ElementaryType == "minjerk":
    #         zerovec = np.zeros(shape=np.shape(self.y0))
    #         if dy0 is not None and np.shape(dy0)[0] == np.shape(self.y0)[0]:
    #             dy0 = dy0
    #         else:
    #             dy0 = zerovec
    #
    #         if ddy0 is not None and np.shape(ddy0)[0] == np.shape(self.y0)[0]:
    #             ddy0 = ddy0
    #         else:
    #             ddy0 = zerovec
    #
    #         if dg is not None and np.shape(dg)[0] == np.shape(self.y0)[0]:
    #             dg = dg
    #         else:
    #             dg = zerovec
    #
    #         if ddg is not None and np.shape(ddg)[0] == np.shape(self.y0)[0]:
    #             ddg = ddg
    #         else:
    #             ddg = zerovec
    #
    #         self.h_params = self.get_min_jerk_params(self.y0 , self.g, dy0=dy0, dg=dg, ddy0=ddy0, ddg=ddg)
    #     else:
    #         self.h_params = np.transpose(np.stack([self.g, self.y0 - self.g]))

    def roll(self, y0, g, n_samples=None):
        """
        reproduce the trajectory given start point y0 (dim, ) and end point g (dim, ), return traj (n_samples, dim)
        """
        n_samples = self.n_samples if n_samples is None else n_samples
        can_values = self.can_sys(1, 0, n_samples)

        if self.elementary_type == "minjerk":
            dv = np.zeros(y0.shape)
            self.h_params = self.get_min_jerk_params(y0, g, dv, dv, dv, dv)
        else:
            self.h_params = np.stack([g, y0 - g])

        linear_traj = self.linear_traj(can_values)

        psi = self.__Psi__(can_values)  # (T, K)
        print('psi_shape:',psi.shape, 'kernel_weights_shape:',self.kernel_weights.shape)
        traj = linear_traj + np.einsum("ij,jk->ik", psi, self.kernel_weights)

        time_stamp = 1 - np.expand_dims(can_values, 1)
        return np.concatenate([time_stamp, traj], axis=1)

    def get_target(self, t):
        action = np.transpose(self.h(1-t)) + self.get_position(t)
        return action

    @staticmethod
    def can_sys(t0, t1, n_sample):
        """
        return the sampled values of linear decay canonical system

        Args:
            t0: start time point
            t1: end time point
            n_sample: number of samples
        """
        return np.linspace(t0, t1, n_sample)

    @staticmethod
    def get_min_jerk_params(y0, g, dy0, dg, ddy0, ddg):
        b = np.stack([y0, dy0, ddy0, g, dg, ddg])
        A = np.array(
            [[1, 1, 1, 1, 1, 1], [0, 1, 2, 3, 4, 5], [0, 0, 2, 6, 12, 20], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
             [0, 0, 2, 0, 0, 0]])

        return np.linalg.solve(A, b)


if __name__ == '__main__':
    traj_files = [
        "/home/gao/projects/control/visual-imitation-learning/data/demo/real/insertion_1_demo_2802_black_background/constraints/traj_00/demo_00.csv",
        "/home/gao/projects/control/visual-imitation-learning/data/demo/real/insertion_1_demo_2802_black_background/constraints/traj_01/demo_00.csv",
        "/home/gao/projects/control/visual-imitation-learning/data/demo/real/insertion_1_demo_2802_black_background/constraints/traj_02/demo_00.csv"
    ]
    trajs = np.array([np.loadtxt(f, delimiter=',') for f in traj_files])
    print(trajs.shape)


    start = np.array([ 109.20813747, -159.67668235,  -14.46389848])
    # start = trajs[0, 0, 1] - 100, trajs[0, 0, 2] - 50, trajs[0, 0, 3]
    goal = np.array([9.298113053909816, -1.8103986074593479, 6.167278413806116])
    print(start, trajs[0, 0, :])
    vmp = VMP(3, kernel_num=50, elementary_type='linear', use_out_of_range_kernel=False)
    linear_traj_raw = vmp.train(trajs[[0]])
    reproduced, linear_traj = vmp.roll(start, goal, 50)
    print(reproduced.shape, linear_traj.shape)


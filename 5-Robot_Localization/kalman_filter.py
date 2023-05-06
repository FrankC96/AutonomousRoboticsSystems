import numpy as np
import sympy as sp


class KalmanFilter:
    def __init__(self,
                 x,
                 u,
                 meas,
                 P,
                 Q,
                 R,
                 dt,
                 lm):

        self.x = x
        self.u = u
        self.meas = meas

        self.P = P
        self.Q = Q
        self.R = R

        self.K = np.empty([3, 2])

        self.dt = dt
        self.lm = lm

        self.x_prime = np.zeros([1, 3])
        self.P_prime = np.zeros([3, 3])

    def compute_loc(self):

        r1 = np.sqrt((self.x[0] - self.lm[0][0])**2 + (self.x[1] - self.lm[0][1])**2)
        phi1 = np.arctan2(self.lm[0][1] - self.x[1], self.lm[0][0] - self.x[0]) - self.x[2]

        r2 = np.sqrt((self.x[0] - self.lm[1][0])**2 + (self.x[1] - self.lm[1][1])**2)
        phi2 = np.arctan2(self.lm[1][1] - self.x[1], self.lm[1][0] - self.x[0]) - self.x[2]

        r3 = np.sqrt((self.x[0] - self.lm[2][0])**2 + (self.x[1] - self.lm[2][1])**2)
        phi3 = np.arctan2(self.lm[2][1] - self.x[1], self.lm[2][0] - self.x[0]) - self.x[2]

        x1_bar = self.x[0] + r1 * np.cos(phi1 + self.x[2])
        y1_bar = self.x[1] + r1 * np.sin(phi1 + self.x[2])
        x2_bar = self.x[0] + r2 * np.cos(phi2 + self.x[2])
        y2_bar = self.x[1] + r2 * np.sin(phi2 + self.x[2])
        x3_bar = self.x[0] + r3 * np.cos(phi3 + self.x[2])
        y3_bar = self.x[1] + r3 * np.sin(phi3 + self.x[2])

        # self.meas = np.array([np.average((x1_bar, x2_bar, x3_bar)), np.average((y1_bar, y2_bar, y3_bar))])
        self.meas = np.array([x1_bar, y1_bar])
    def trackPhone(self):
        x1 = self.lm[0][0]
        x2 = self.lm[0][1]

        y1 = self.lm[1][0]
        y2 = self.lm[1][1]

        x3 = self.lm[2][0]
        y3 = self.lm[2][1]

        r1 = np.sqrt((self.x[0] - x1) ** 2 + (self.x[1] - y1) ** 2)
        r2 = np.sqrt((self.x[0] - x2) ** 2 + (self.x[1] - y2) ** 2)
        r3 = np.sqrt((self.x[0] - x3) ** 2 + (self.x[1] - y3) ** 2)

        A = 2 * x2 - 2 * x1
        B = 2 * y2 - 2 * y1
        C = r1 ** 2 - r2 ** 2 - x1 ** 2 + x2 ** 2 - y1 ** 2 + y2 ** 2
        D = 2 * x3 - 2 * x2
        E = 2 * y3 - 2 * y2
        F = r2 ** 2 - r3 ** 2 - x2 ** 2 + x3 ** 2 - y2 ** 2 + y3 ** 2
        x = (C * E - F * B) / (E * A - B * D)
        y = (C * D - A * F) / (B * D - A * E)

        self.meas = np.array([x, y])

    def predict(self):
        A = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])

        B = np.array([
            [self.dt * np.cos(self.x[2]),       0],
            [-self.dt * np.sin(self.x[2]),       0],
            [0                          , self.dt]
        ])

        self.C = np.array([[1, 0, 0],
                          [0, 1, 0]])

        self.x = A.dot(self.x) + B.dot(self.u)

        self.y = self.C @ self.x

        self.P = A.dot(self.P).dot(A).T + self.R


    def correct(self):
        # self.K = self.P_prime.dot(self.C.T).dot(np.linalg.inv(self.C.dot(self.P_prime).dot(self.C.T) + self.Q))

        self.K = (self.P @ self.C.T) @ np.linalg.inv(self.C @ self.P @ self.C.T + self.Q)
        self.x_prime = self.x + self.K.dot(self.meas - self.C.dot(self.x)).reshape([1, 3])
        self.P_prime = (np.eye(3) - self.K.dot(self.C)).dot(self.P)


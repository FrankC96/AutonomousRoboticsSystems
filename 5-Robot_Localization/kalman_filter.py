import numpy as np


class KalmanFilter:
    def __init__(self,
                 x,
                 u,
                 meas,
                 P,
                 Q,
                 R,
                 dt):

        self.x = x
        self.u = u

        self.P = P
        self.Q = Q
        self.R = R

        self.K = np.empty([3, 2])

        self.dt = dt
        self.x_prime = []
        self.P_prime = []

    def predict(self):
        A = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        B = np.array([
            [self.dt * np.cos(self.x[2]),       0],
            [self.dt * np.sin(self.x[2]),       0],
            [0                          , self.dt]
        ])

        self.C = np.array([1, 1, 0])

        self.x_prime = A.dot(self.x) + B.dot(self.u)
        self.P_prime = A.dot(self.P).dot(A).T + self.R

    def correct(self):
        self.K = self.P_prime.dot(self.C.T).dot(np.linalg.inv(self.C.dot(self.P_prime).dot(self.C.T) + self.Q))
        self.x = self.x_prime + self.K.dot(self.meas - self.C.dot(self.x_prime))
        self.P = (np.eye(3) - self.K.dot(self.C)).dot(self.P_prime)
        print(self.P)


P = np.eye(3)
Q = np.array([
    [0.1, 0],
    [0, 0.1]
])
R = 0.2 * np.eye(3)

kf = KalmanFilter(x=[0, 0, 0], u=[0, 0], meas=0, P=P, Q=Q, R=R, dt=0.3)

for k in range(10):

    kf.predict()
    kf.predict()
    # print(kf.P)

# print(kf.K)

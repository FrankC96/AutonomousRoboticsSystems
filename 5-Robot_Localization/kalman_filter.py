import numpy as np

class KalmanFilter:
    def __init__(self,
                 x,
                 P):

        self.x = x
        self.P = P

    def predict(self):
        A = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        B = np.array([])
        self.x =
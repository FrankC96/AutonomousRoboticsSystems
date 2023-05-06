import numpy as np
import matplotlib.pyplot as plt

from kalman_filter import KalmanFilter


P = 0.1 * np.eye(3)
Q = 100 * np.eye(2)
R = 0.1 * np.eye(3)

kf = KalmanFilter(x=[0, 0, 0], u=[10, 1], meas=0, P=P, Q=Q, R=R, dt=0.05, lm=(2, 2))

for k in range(1000):
    kf.predict()
    if k == k:
        print(f"Measure at timestep {k} with {kf.x_prime}.")
        # kf.meas = kf.x[0] + 100

        kf.compute_loc()
        kf.correct()

    plt.figure(1)
    plt.plot(k, kf.x[0], "kx")
    plt.plot(k, kf.x_prime[0], "rx")

    plt.figure(2)
    plt.plot(k, kf.x[1], "kx")
    plt.plot(k, kf.x_prime[1], "rx")

    plt.figure(3)
    plt.plot(k, kf.x[2], "kx")
    plt.plot(k, kf.x_prime[2], "rx")

    plt.pause(0.05)

plt.show()
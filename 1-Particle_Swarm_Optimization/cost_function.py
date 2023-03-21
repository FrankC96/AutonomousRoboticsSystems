import numpy as np
import control.matlab as ctr
import matplotlib.pyplot as plt


def pid_eval(x):

    M = 1
    g = 9.81
    l = 0.5
    J = 0.001
    A = np.array([[0, 1], [(M*g*l)/(2*J), 0]])
    B = np.array([[0], [1/(J)]])
    C = np.array([1, 0])
    D = np.array([0])
    # A = np.array(   [[-0.0015,   -0.0013,    0.2504,         0],
    #                 [-0.0001,   -0.0122,   -3.4096,         0],
    #                 [0.0000,  -0.0003,   -0.0461,         0],
    #                 [0,         0,    1.0000,         0]])
    #
    # B = np.array([[0.0011],
    #               [0.0238],
    #               [-0.0006],
    #               [0]])
    #
    # C = np.array([0, 0, 0, 1])
    #
    # D = np.array([0])

    Kp = x[0]
    Ki = x[1]
    Kd = x[2]

    stb = 1000
    if np.shape(Kp) == ():
        if Kp > stb:
            print("clipping: ", Kp)
            Kp = stb
            print("to: ", Kp)
        elif Kp < -stb:
            print("clipping: ", Kp)
            Kp = -stb
            print("to: ", Kp)

        if Ki > stb:
            print("clipping: ", Ki)
            Ki = stb
            print("to: ", Ki)
        elif Ki < -stb:
            print("clipping: ", Ki)
            Ki = -stb
            print("to: ", Ki)
        if Kd > stb:
            print("clipping: ", Kd)
            Kd = stb
            print("to: ", Kd)
        elif Kd < -stb:
            print("clipping: ", Kd)
            Kd = -stb
            print("to: ", Kd)

    sys_ss = ctr.ss(A, B, C, D)
    sys_tf = ctr.ss2tf(sys_ss)

    s = ctr.tf('s')
    con_pid = (Kp*s + Ki + Kd * s**2)/s

    H = ctr.feedback(sys_tf * con_pid, 1)
    t_sim = 450

    t = np.arange(t_sim)
    u = np.pi/10 * np.ones(t_sim)
    x0 = np.array([0, 0, 0])

    yout, T, xout = ctr.lsim(H, u, t, x0)

    error = np.sum((u - yout) ** 2) + 100 * np.sum((u[-1:100] - yout[t_sim:-1]) ** 2)

    return error  # RETURN ALSO yout

def cost(x):

    stable_thresh = 10
    x1 = x[0]
    x2 = x[1]

    if np.shape(x1) == ():
        if x1 > stable_thresh:
            x1 = stable_thresh
        elif x1 < -stable_thresh:
            x1 = -stable_thresh

        if x2 > stable_thresh:
            x2 = stable_thresh
        elif x2 < -stable_thresh:
            x2 = -stable_thresh

    eggcrate = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1/2 + 47))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))
    beale = (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2
    matyas = 0.26*(x1**2 + x2**2) + 0.48*x1*x2
    booth = (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2
    thefuck = x1**2 + x2**2
    eggholder = -(x2 + 47) * np.sin(np.sqrt(np.abs(x1/2 + (x2 + 47)))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))
    mccormick = np.sin(x1 + x2) + (x1 - x2)**2 - 1.5*x1 + 2.5*x2 + 1
    rosenbrock = (0-x1)**2 + 100 * (x2 - x1**2)**2
    rastrigin = 20 + (x1**2 - 10*np.cos(2*np.pi*x1) + x2**2 - 10*np.cos(2*np.pi*x2))
    return rastrigin
import daceypy_import_helper  # noqa: F401

from typing import Callable, Type

import numpy as np

from daceypy import DA, RK, array, integrator
from daceypy._integrator import PicardLindelof

mu = 398600.0  # km^3/s^2


def RK78(Y0: array, X0: float, X1: float, f: Callable[[array, float], array]):

    Y0 = Y0.copy()

    N = len(Y0)

    H0 = 0.001
    HS = 0.1
    H1 = 100.0
    EPS = 1.e-12
    BS = 20 * EPS

    Z = array.zeros((N, 16))
    Y1 = array.zeros(N)

    VIHMAX = 0.0

    HSQR = 1.0 / 9.0
    A = np.zeros(13)
    B = np.zeros((13, 12))
    C = np.zeros(13)
    D = np.zeros(13)

    A = np.array([
        0.0, 1.0/18.0, 1.0/12.0, 1.0/8.0, 5.0/16.0, 3.0/8.0, 59.0/400.0,
        93.0/200.0, 5490023248.0/9719169821.0, 13.0/20.0,
        1201146811.0/1299019798.0, 1.0, 1.0,
    ])

    B[1, 0] = 1.0/18.0
    B[2, 0] = 1.0/48.0
    B[2, 1] = 1.0/16.0
    B[3, 0] = 1.0/32.0
    B[3, 2] = 3.0/32.0
    B[4, 0] = 5.0/16.0
    B[4, 2] = -75.0/64.0
    B[4, 3] = 75.0/64.0
    B[5, 0] = 3.0/80.0
    B[5, 3] = 3.0/16.0
    B[5, 4] = 3.0/20.0
    B[6, 0] = 29443841.0/614563906.0
    B[6, 3] = 77736538.0/692538347.0
    B[6, 4] = -28693883.0/1125000000.0
    B[6, 5] = 23124283.0/1800000000.0
    B[7, 0] = 16016141.0/946692911.0
    B[7, 3] = 61564180.0/158732637.0
    B[7, 4] = 22789713.0/633445777.0
    B[7, 5] = 545815736.0/2771057229.0
    B[7, 6] = -180193667.0/1043307555.0
    B[8, 0] = 39632708.0/573591083.0
    B[8, 3] = -433636366.0/683701615.0
    B[8, 4] = -421739975.0/2616292301.0
    B[8, 5] = 100302831.0/723423059.0
    B[8, 6] = 790204164.0/839813087.0
    B[8, 7] = 800635310.0/3783071287.0
    B[9, 0] = 246121993.0/1340847787.0
    B[9, 3] = -37695042795.0/15268766246.0
    B[9, 4] = -309121744.0/1061227803.0
    B[9, 5] = -12992083.0/490766935.0
    B[9, 6] = 6005943493.0/2108947869.0
    B[9, 7] = 393006217.0/1396673457.0
    B[9, 8] = 123872331.0/1001029789.0
    B[10, 0] = -1028468189.0/846180014.0
    B[10, 3] = 8478235783.0/508512852.0
    B[10, 4] = 1311729495.0/1432422823.0
    B[10, 5] = -10304129995.0/1701304382.0
    B[10, 6] = -48777925059.0/3047939560.0
    B[10, 7] = 15336726248.0/1032824649.0
    B[10, 8] = -45442868181.0/3398467696.0
    B[10, 9] = 3065993473.0/597172653.0
    B[11, 0] = 185892177.0/718116043.0
    B[11, 3] = -3185094517.0/667107341.0
    B[11, 4] = -477755414.0/1098053517.0
    B[11, 5] = -703635378.0/230739211.0
    B[11, 6] = 5731566787.0/1027545527.0
    B[11, 7] = 5232866602.0/850066563.0
    B[11, 8] = -4093664535.0/808688257.0
    B[11, 9] = 3962137247.0/1805957418.0
    B[11, 10] = 65686358.0/487910083.0
    B[12, 0] = 403863854.0/491063109.0
    B[12, 3] = - 5068492393.0/434740067.0
    B[12, 4] = -411421997.0/543043805.0
    B[12, 5] = 652783627.0/914296604.0
    B[12, 6] = 11173962825.0/925320556.0
    B[12, 7] = -13158990841.0/6184727034.0
    B[12, 8] = 3936647629.0/1978049680.0
    B[12, 9] = -160528059.0/685178525.0
    B[12, 10] = 248638103.0/1413531060.0

    C = np.array([
        14005451.0/335480064.0, 0.0, 0.0, 0.0, 0.0, -59238493.0/1068277825.0,
        181606767.0/758867731.0, 561292985.0/797845732.0,
        -1041891430.0/1371343529.0, 760417239.0/1151165299.0,
        118820643.0/751138087.0, -528747749.0/2220607170.0, 1.0/4.0,
    ])

    D = np.array([
        13451932.0/455176623.0, 0.0, 0.0, 0.0, 0.0, -808719846.0/976000145.0,
        1757004468.0/5645159321.0, 656045339.0/265891186.0,
        -3867574721.0/1518517206.0, 465885868.0/322736535.0,
        53011238.0/667516719.0, 2.0/45.0, 0.0,
    ])

    Z[:, 0] = Y0

    H = abs(HS)
    HH0 = abs(H0)
    HH1 = abs(H1)
    X = X0
    RFNORM = 0.0
    ERREST = 0.0

    while X != X1:

        # compute new stepsize
        if RFNORM != 0:
            H = H * min(4.0, np.exp(HSQR * np.log(EPS / RFNORM)))
        if abs(H) > abs(HH1):
            H = HH1
        elif abs(H) < abs(HH0) * 0.99:
            H = HH0
            print("--- WARNING, MINIMUM STEPSIZE REACHED IN RK")

        if (X + H - X1) * H > 0:
            H = X1 - X

        for j in range(13):

            for i in range(N):

                Y0[i] = 0.0
                # EVALUATE RHS AT 13 POINTS
                for k in range(j):
                    Y0[i] = Y0[i] + Z[i, k + 3] * B[j, k]

                Y0[i] = H * Y0[i] + Z[i, 0]

            Y1 = f(Y0, X + H * A[j])

            for i in range(N):
                Z[i, j + 3] = Y1[i]

        for i in range(N):

            Z[i, 1] = 0.0
            Z[i, 2] = 0.0
            # EXECUTE 7TH,8TH ORDER STEPS
            for j in range(13):
                Z[i, 1] = Z[i, 1] + Z[i, j + 3] * D[j]
                Z[i, 2] = Z[i, 2] + Z[i, j + 3] * C[j]

            Y1[i] = (Z[i, 2] - Z[i, 1]) * H
            Z[i, 2] = Z[i, 2] * H + Z[i, 0]

        Y1cons = Y1.cons()

        # ESTIMATE ERROR AND DECIDE ABOUT BACKSTEP
        RFNORM = np.linalg.norm(Y1cons, np.inf)  # type: ignore
        if RFNORM > BS and abs(H / H0) > 1.2:
            H = H / 3.0
            RFNORM = 0
        else:
            for i in range(N):
                Z[i, 0] = Z[i, 2]
            X = X + H
            VIHMAX = max(VIHMAX, H)
            ERREST = ERREST + RFNORM

    Y1 = Z[:, 0]

    return Y1


def TBP(x: array, t: float) -> array:
    pos: array = x[:3]
    vel: array = x[3:]
    r = pos.vnorm()
    acc: array = -mu * pos / (r ** 3)
    dx = vel.concat(acc)
    return dx


def TBP_time(x: array, tau: float, t0: DA, tf: DA) -> array:
    # input time tau is normalized. To retrieve t: tau*(tf-t0)
    # RHS of ODE must be multiplied by (tf-t0) to scale

    # t is computed but useless in case of autonomous dynamics
    t = tau * (tf - t0)
    pos: array = x[:3]
    vel: array = x[3:]
    r = pos.vnorm()
    acc: array = -mu * pos / (r ** 3)
    dx = (tf-t0)*(vel.concat(acc))
    return dx


class TBP_integrator_time(integrator):
    def __init__(self, RK: RK.RKCoeff = RK.RK78(), stateType: Type = np.ndarray):
        super(TBP_integrator_time, self).__init__(RK, stateType)

        # auxiliary parameters to store time variation, may be included in state
        # this would complicate the RHS of the ODE but would simplify this class
        self.T0 = None
        self.TF = None

    def f(self, x, t):
        return TBP_time(x,t, self.T0, self.TF)


class TBP_integrator_float(integrator):
    def __init__(self, RK: RK.RKCoeff = RK.RK78(), stateType: Type = np.ndarray):
        super(TBP_integrator_float, self).__init__(RK, stateType)

    def f(self, x, t):
        return TBP(x,t)


def main():

    DA.init(8, 7)
    DA.setEps(1e-32)

    # Set initial conditions
    ecc = 0.5

    x0 = array.zeros(6)
    x0[0] += 6678.0  # 300 km altitude
    x0[4] += np.sqrt(mu / 6678.0 * (1 + ecc))

    # integrate for half the orbital period
    a = 6678 / (1 - ecc)

    # application of time expansion with user defined integrator:
    # Method taken from Sect. 6 of https://doi.org/10.1007/s10569-010-9283-5.
    # The idea is to define a normalized time that varies from 0 (at t = t0)
    # to 1 (at t = tf).
    # This normalization requires modification of the RHS of the ODE but
    # transforms initial and final time into parameters, hence removing the
    # need of a custom integrator capable of dealing with DA time.

    # a lambda function needs to be created to fix t0 and tf for the propagation.
    RHS = lambda x, tau : TBP_time(x, tau, 0.0, np.pi * np.sqrt(a**3 / mu) + DA(7))
    # integration is carried out in normalized time between 0 and 1:
    xf = RK78(x0, 0.0, 1.0, RHS)

    # Alternatively, one can use the modular propagator provided in daceypy:
    # creation of instance of propagator class with correct dynamics
    propagator_78 = TBP_integrator_time(RK.RK78(), array)

    # custom properties to store initial and final time as DA variables:
    # they will be used for scaling the dynamics as previously done with
    # the lambda function.
    propagator_78.T0 = 0.0
    propagator_78.TF = np.pi * np.sqrt(a**3 / mu) + DA(7)

    # once again the integration is carried out in normalized time between 0 and 1:
    propagator_78.loadTime(0.0, 1.0)
    propagator_78.loadTol(20*1e-12, 1e-12)
    propagator_78.loadStepSize()
    xf_modular=propagator_78.propagate(x0, 0.0, 1.0)

    # comparison with integrator of only constant part and non scaled time:
    propagator_78f=TBP_integrator_float(RK.RK78(), array)
    propagator_78f.loadTime(0.0,  np.pi * np.sqrt(a**3 / mu))
    propagator_78f.loadTol(20*1e-12, 1e-12)
    propagator_78f.loadStepSize()
    xf_modular_float = propagator_78f.propagate(x0, 0.0, np.pi * np.sqrt(a**3 / mu))

    print("Final error of constant part with and without time scaling: \n",
           xf_modular.cons()-xf_modular_float)

    # Comparison with expansion at final time computed with picard lindelof operator:
    # NB: only possible at final time

    xf_PL = PicardLindelof(array(xf_modular_float), 7, np.pi * np.sqrt(a**3 / mu), TBP)

    print("Error of final time expansion (integrator vs Picard-Lindelof): \n",
           xf_modular - xf_PL)


if __name__ == "__main__":
    main()

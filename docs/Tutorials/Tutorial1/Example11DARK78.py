import daceypy_import_helper  # noqa: F401

from typing import Callable, Type
import numpy as np
from daceypy import DA, array, RK, integrator, integrator_optimized, DA_utils
import time
mu = 398600.4418  # [km^3/s^2] Earth gravitational parameter, used by TBP()


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

class TBP_integrator(integrator):
    def __init__(self, RK: RK.RKCoeff = RK.RK78(),  stateType: Type = array):
        super(TBP_integrator, self).__init__(RK, stateType)

    def f(self, x, t):
        return TBP(x,t)


class TBP_integrator_optimized(integrator_optimized):
    def __init__(self, RK: RK.RKCoeff = RK.RK78(),  stateType: Type = array, DA_type = 'DA_direct'):
        super(TBP_integrator_optimized, self).__init__(RK, stateType, DA_type)

    def f(self, x, t):
        return TBP(x,t)


def build_optimized_propagator(t0, tf, DA_type, tol=1e-16, integrator=None):
    """Factory to avoid repeating the loadTime/loadTol/loadStepSize boilerplate
    every time a TBP_integrator_optimized instance is created."""
    integrator = integrator or RK.RK78()
    prop = TBP_integrator_optimized(integrator, array, DA_type=DA_type)
    prop.loadTime(t0, tf)
    prop.loadTol(tol, tol)
    prop.loadStepSize()
    return prop


def main():
    # -------------------------------------------------------------------------
    # STEPWISE vs DIRECT DA PROPAGATION — CONCEPTUAL DIFFERENCE
    # -------------------------------------------------------------------------
    # DA_direct:
    #   - Every DA map is referenced to the SAME initial time t0.
    #   - For each evaluation time t in t_eval, the integrator builds a map t0 -> t.
    #   - Best suited for sensitivity analysis, uncertainty propagation,
    #     or Monte Carlo sampling around the SAME initial condition.
    #   - Accuracy may degrade over long propagation intervals if
    #     non-linearities grow significantly.
    #
    # DA_stepwise:
    #   - DA maps are constructed LOCALLY between consecutive time segments.
    #   - The DA expansion is reset at each intermediate time, increasing accuracy.
    #   - Does NOT directly yield a single map t0 -> tf — must be chained.
    #
    # ASCII illustration:
    #   DIRECT:     t0 --------> t1
    #               t0 ---------------------> t2
    #
    #   STEPWISE:   t0 --------> t1 --------> t2
    # -------------------------------------------------------------------------
    # Initialize DA and orbital parameters
    # -------------------------------------------------------------------------

    DA.init(2, 6)  # 2nd order, 6 variables

    ecc = 0.5
    x0 = array.identity(6)
    x0[0] += 6678.0
    x0[4] += np.sqrt(mu / 6678.0 * (1 + ecc))

    a = 6678.0 / (1 - ecc)
    T = 2 * np.pi * np.sqrt(a**3 / mu)

    print(f"Orbital period: {T:.2f} s, Semi-major axis: {a:.2f} km, Eccentricity: {ecc}")

    t_eval = [0.0, 500.0, 600.0]
    t0, tf = t_eval[0], t_eval[-1]

    # -------------------------------------------------------------------------
    # Test 1: Old (non-DA) integrator — reference solution
    # -------------------------------------------------------------------------
    print("\n=== TEST 1: OLD TBP_INTEGRATOR ===")
    propagator_old = TBP_integrator(RK.RK78(), array)
    propagator_old.loadTime(t0, tf)
    propagator_old.loadTol(1e-16, 1e-16)
    propagator_old.loadStepSize()

    start_old = time.perf_counter()
    xf_old = propagator_old.propagate(x0, t0, tf)
    time_old = time.perf_counter() - start_old
    print(f"Old computational time: {time_old:.6f} s")

    # -------------------------------------------------------------------------
    # Test 2: Optimized DA_direct integrator, evaluated only at t_eval
    # -------------------------------------------------------------------------
    print("\n=== TEST 2: OPTIMIZED TBP_INTEGRATOR (DA_direct) ===")
    propagator_direct = build_optimized_propagator(t0, tf, "DA_direct")

    start_opt = time.perf_counter()
    xf_direct = propagator_direct.propagate(x0, t_eval)
    time_opt = time.perf_counter() - start_opt
    print(f"Optimised computational time: {time_opt:.6f} s")

    # -------------------------------------------------------------------------
    # Test 3: Compare old vs optimized final state
    # (use .cons() to pull the constant/nominal part out of the DA object
    #  before comparing it against the plain numeric result from Test 1)
    # -------------------------------------------------------------------------
    print("\n=== TEST 3: COMPARISON OLD vs DA_direct (final state) ===")
    map_diff_final = xf_direct[-1][0] - xf_old[0]
    print(f"Difference between final states (old vs optimized):\n {map_diff_final}")

    # -------------------------------------------------------------------------
    # Test 4: Optimized DA_direct integrator, saving every integration step
    # -------------------------------------------------------------------------
    print("\n=== TEST 4: OPTIMIZED TBP_INTEGRATOR (DA_direct) - all integrator steps ===")
    propagator_direct_all = build_optimized_propagator(t0, tf, "DA_direct")

    start_opt = time.perf_counter()
    xf_direct_all, times_all = propagator_direct_all.propagate(x0, t_eval, save_all_times=True)
    time_opt = time.perf_counter() - start_opt
    print(f"Optimised computational time: {time_opt:.6f} s")

    map_diff_final_all = xf_direct_all[-1][0] - xf_old[0]
    print(f"Difference between final states (old vs optimized, all-steps map):\n {map_diff_final_all}")

    # -------------------------------------------------------------------------
    # Test 5: Stepwise propagation vs chained direct propagation
    # -------------------------------------------------------------------------
    print("\n=== TEST 5: STEP-WISE PROPAGATION ===")

    propagator_step = build_optimized_propagator(t0, tf, "DA_stepwise")
    xf_step = propagator_step.propagate(x0, t_eval)

    # xf_direct[-2] is the state at t_eval[-2]; xf_direct_all[-2] is merely the
    # penultimate accepted integrator step, which does not generally land on
    # t_eval[-2] = 500 s — restarting the chain from xf_direct_all would start
    # at the wrong time.
    x_restart = xf_direct[-2].cons() + array.identity(6)
    propagator_chain = build_optimized_propagator(t_eval[-2], t_eval[-1], "DA_direct")
    xf_chain = propagator_chain.propagate(x_restart, [t_eval[-2], t_eval[-1]])

    map_diff_chain = xf_step[-1][0] - xf_chain[-1][0]
    print(f"\nDifference between direct optimized and chained stepwise: \n {map_diff_chain}")

    # -------------------------------------------------------------------------
    # Test 6: Extract Taylor terms of the DA map at the final time, then
    # write a perturbed tensor back into a copy of the DA state and verify
    # the perturbation actually reached the DA object (not just the dict).
    # -------------------------------------------------------------------------
    print("\n=== TEST 6: EXTRACTION AND RE-ASSIGNMENT OF DA TAYLOR TERMS ===")
    # Use xf_direct (map evaluated exactly at t_eval), consistent with the
    # rest of the analysis, rather than the reassigned all-steps array.
    maps = DA_utils.extract_map(xf_direct, max_order=2)

    print("\nZeroth-order term (nominal final state):")
    print(maps[-1]["Taylor_order_0"])

    print("\nFirst-order term (State Transition Matrix - STM):")
    print(maps[-1]["Taylor_order_1"])

    print("\nSecond-order term (Hessian - nonlinear sensitivities):")
    print(maps[-1]["Taylor_order_2"])

    # Perturb one STM entry in the extracted tensor, then assign the modified
    # map onto a *copy* of xf_direct (xf_direct itself is still needed below).
    perturbation = 1.234e-3
    original_entry = maps[-1]["Taylor_order_1"][0, 0]
    maps[-1]["Taylor_order_1"][0, 0] = original_entry + perturbation

    xf_perturbed = [x.copy() for x in xf_direct]
    DA_utils.assign_taylor_to_da(maps, xf_perturbed)
    maps_post = DA_utils.extract_map(xf_perturbed, max_order=2)
    reassigned_entry = maps_post[-1]["Taylor_order_1"][0, 0]

    print(f"\nSTM[0, 0] before perturbation: {original_entry}")
    print(f"STM[0, 0] after assign_taylor_to_da: {reassigned_entry}")
    assert abs(reassigned_entry - (original_entry + perturbation)) < 1e-9, (
        "assign_taylor_to_da did not write the perturbed coefficient back "
        "into the DA object"
    )
    print("assign_taylor_to_da correctly mutated the DA state")

    # -------------------------------------------------------------------------
    # Test 7: Evaluate DA map with a small displacement
    # -------------------------------------------------------------------------
    Deltax0 = np.array([1.0, -1.0, 0, 0, 0, 0])
    xf_displaced = xf_direct[-1].eval(Deltax0)

    print(f"\n=== TEST 7: DA map evaluation with Deltax0 = {Deltax0} ===")
    print(f"  Position: {xf_displaced[0]:.6f} km")
    print(f"  Velocity: {xf_displaced[3]:.6f} km/s")

if __name__ == "__main__":
    main()

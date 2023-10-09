import daceypy_import_helper  # noqa: F401

import time
from inspect import getsourcefile
from pathlib import Path
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from daceypy import ADS, DA, array

# setup better images (tex fonts)
plt.rcParams.update({
    "figure.max_open_warning": 0,
    "font.family": "Helvetica",
    "mathtext.fontset": "cm",
})

sourcefile = getsourcefile(lambda: 0)
assert sourcefile is not None
thisfolder = Path(sourcefile).resolve().parent

mu = 1.0  # km^3/s^2

def RK78(Y0: array, X0: float, X1: float, f: Callable[[array, float], array]) -> array:
    """
    Propagate using RK78.
    """
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
    """
    2D Two Body Problem dynamics.
    """
    pos: array = x[:2]
    vel: array = x[2:]
    r = pos.vnorm()
    acc: array = -mu * pos / (r ** 3)
    dx = vel.concat(acc)
    return dx


# evaluation functions for ADS
def base_propagation(domain_0: ADS, t0: float, tf: float) -> ADS:
    """
    Base ADS propagation function.
    """
    x0 = domain_0.box
    xf = RK78(x0, t0, tf, TBP)
    return ADS(domain_0.box, domain_0.nsplit, xf)


def advanced_propagation(domain_0: ADS, t0: float, tf: float) -> ADS:
    """
    Advanced ADS propagation function.
    """
    x0 = domain_0.manifold
    xf = RK78(x0, t0, tf, TBP)
    return ADS(domain_0.box, domain_0.nsplit, xf)


# functions for figures
def figure_1(XF: NDArray[np.double], Ns: float, tgrid: NDArray[np.double], instants: List[int] = [0, 1, 2, 4, 6, 9, 12, 16]) -> plt.Axes:
    _, ax = plt.subplots()
    for j in instants:
        for i in range(4):
            ax.plot(XF[0,j, Ns * i : Ns * (i + 1)], XF[1, j, Ns * i : Ns * (i + 1)], color='black')
            label = "{:.2f}".format(tgrid[j])
            ax.annotate("$t_i = $ " + label, # this is the text
                        (XF[0,j,0],XF[1,j,0]), # these are the coordinates to position the label
                        textcoords="offset points", # how to position the text
                        xytext=(0,10), # distance from text to points (x,y)
                        ha='center') # horizontal alignment can be left, right or center
    ax.plot(0, 0, marker="o", markeredgecolor="black", markerfacecolor="none")
    ax.grid('minor',  linestyle=':')
    ax.set_xlabel('x (-)')
    ax.set_ylabel('y (-)')
    return ax

def figure_2(XF: NDArray[np.double], XF1: NDArray[np.double], XF2: NDArray[np.double], XF6: NDArray[np.double],
             Ns: int, analysis_time: int = 16) -> plt.Axes:
    _ , ax = plt.subplots()
    for i in range(4):
        ax.plot(XF[0, analysis_time, Ns * i : Ns * (i + 1)], XF[1, analysis_time, Ns * i : Ns * (i + 1)], color='black')
        ax.plot(XF1[0, analysis_time, Ns * i : Ns * (i + 1)], XF1[1, analysis_time, Ns * i : Ns * (i + 1)], color='black',  linestyle='--')
        ax.plot(XF2[0, analysis_time, Ns * i : Ns * (i + 1)], XF2[1, analysis_time, Ns * i : Ns * (i + 1)], color='black', linestyle=':')
        ax.plot(XF6[0, analysis_time, Ns * i : Ns * (i + 1)], XF6[1, analysis_time, Ns * i : Ns * (i + 1)], color='black',  linestyle='-.')
        ax.legend(("Nominal","1st order", "2nd order", "6th order"),
          shadow=True, loc="best", handlelength=1.5, fontsize=16)
    ax.grid('minor',  linestyle=':')
    ax.set_xlabel('x (-)')
    ax.set_ylabel('y (-)')
    return ax

def figure_3(XF: NDArray[np.double], XF6: NDArray[np.double], XF14: NDArray[np.double], Ns: int, analysis_time: int = 16) -> plt.Axes:
    _ , ax = plt.subplots(nrows=1,ncols=2)

    for i in range(4):
        ax[0].plot(XF[0, analysis_time, Ns * i : Ns * (i + 1)], XF[1, analysis_time, Ns * i : Ns * (i + 1)], color='black')
        ax[1].plot(XF[0, analysis_time, Ns * i : Ns * (i + 1)], XF[1, analysis_time, Ns * i : Ns * (i + 1)], color='black')
        ax[0].plot(XF6[0, analysis_time, Ns * i : Ns * (i + 1)], XF6[1, analysis_time, Ns * i : Ns * (i + 1)], color='black',  linestyle='-.')
        ax[1].plot(XF14[0, analysis_time, Ns * i : Ns * (i + 1)], XF14[1, analysis_time, Ns * i : Ns * (i + 1)], color='black',  linestyle=':')
        ax[0].legend(("Nominal","6th order"),
          shadow=True, loc="best", handlelength=1.5, fontsize=16)
        ax[1].legend(("Nominal","14th order"),
          shadow=True, loc="best", handlelength=1.5, fontsize=16)
    ax[0].grid('minor',  linestyle=':')
    ax[0].set_xlabel('x (-)')
    ax[0].set_ylabel('y (-)')
    ax[1].grid('minor',  linestyle=':')
    ax[1].set_xlabel('x (-)')
    ax[1].set_ylabel('y (-)')

    return ax


def figure_4(
    final_lists: List[List[ADS]], XF: NDArray[np.double], XF14: NDArray[np.double],
    Ns: int, perimeter_norm: NDArray[np.double], time_analysis: NDArray[np.int32],
) -> plt.Axes:

    final_map_list=[]
    final_domain_list=[]
    for i in range(len(time_analysis)): # first interesting one is 16?
        final_manifold = np.zeros((4,perimeter_norm.shape[0],len(final_lists[i])))
        final_domain = np.zeros((4,perimeter_norm.shape[0], len(final_lists[i])))
        for j in range(len(final_lists[i])):
            for k in range(perimeter_norm.shape[0]):
                final_manifold[:,k,j] = final_lists[i][j].manifold.eval(perimeter_norm[k,:])
                final_domain[:,k,j] = final_lists[i][j].box.eval(perimeter_norm[k,:])
        final_map_list.append(final_manifold)
        final_domain_list.append(final_domain)

    for i in range(len(time_analysis)):
        fig , ax = plt.subplots(nrows=1,ncols=2)
        for j in range(final_map_list[i].shape[2]):
            for k in range(4):
                ax[0].plot(XF[0,time_analysis[i], Ns * k : Ns * (k + 1)], XF[1,time_analysis[i], Ns * k : Ns * (k + 1)], color='black',  linestyle=':')
                ax[0].plot(XF14[0,time_analysis[i], Ns * k : Ns * (k + 1)], XF14[1,time_analysis[i], Ns * k : Ns * (k + 1)], color='black',  linestyle='--')
                ax[0].plot(final_map_list[i][0, Ns * k : Ns * (k + 1),j],final_map_list[i][1, Ns * k : Ns * (k + 1),j], color='black')
                ax[1].plot(final_domain_list[i][0, Ns * k : Ns * (k + 1),j],final_domain_list[i][1, Ns * k : Ns * (k + 1),j], color='black')

        ax[0].grid('minor',  linestyle=':')
        ax[0].set_xlabel('x (-)')
        ax[0].set_ylabel('y (-)')
        ax[1].grid('minor',  linestyle=':')
        ax[1].set_xlabel('x (-)')
        ax[1].set_ylabel('y (-)')
        ax[0].set_title('mapped domain')
        ax[1].set_title('initial domain')
        ax[0].legend(("Nominal", "14th order", "ADS 14th order"),
          shadow=True, loc="best", handlelength=1.5, fontsize=16)
        fig.suptitle(' Time of analysis : '+ str(time_analysis[i]), fontsize=20)


def figure_5(Ts: int, tgrid: NDArray[np.double], final_lists: List[List[ADS]]) -> plt.Axes:
    nsplit = np.zeros((Ts))
    nsplit[0]= 1

    for i in range(Ts-1):
        nsplit[i+1]=len(final_lists[i])

    _ , ax = plt.subplots()
    ax.plot(tgrid, nsplit)
    ax.grid('minor',  linestyle=':')
    ax.set_xlabel('propagation time (-)')
    ax.set_ylabel('number of domains (-)')

    return ax


def figure_6(
    final_lists: List[List[ADS]], XF: NDArray[np.double], XF14: NDArray[np.double],
    Ns: int, perimeter_norm: NDArray[np.double], time_analysis: NDArray[np.int32],
) -> plt.Axes:

    final_map_list=[]
    final_domain_list=[]
    for i in range(len(time_analysis)): # first interesting one is 16?
        final_manifold = np.zeros((4,perimeter_norm.shape[0],len(final_lists[time_analysis[i]])))
        final_domain = np.zeros((4,perimeter_norm.shape[0], len(final_lists[time_analysis[i]])))
        for j in range(len(final_lists[time_analysis[i]])):
            for k in range(perimeter_norm.shape[0]):
                final_manifold[:,k,j] = final_lists[time_analysis[i]][j].manifold.eval(perimeter_norm[k,:])
                final_domain[:,k,j] = final_lists[time_analysis[i]][j].box.eval(perimeter_norm[k,:])
        final_map_list.append(final_manifold)
        final_domain_list.append(final_domain)

    for i in range(len(time_analysis)):
        fig, ax = plt.subplots(nrows=1,ncols=2)
        for j in range(final_map_list[i].shape[2]):
            for k in range(4):
                ax[0].plot(XF[0,time_analysis[i], Ns * k : Ns * (k + 1)], XF[1,time_analysis[i], Ns * k : Ns * (k + 1)], color='black',  linestyle=':')
                ax[0].plot(XF14[0,time_analysis[i], Ns * k : Ns * (k + 1)], XF14[1,time_analysis[i], Ns * k : Ns * (k + 1)], color='black',  linestyle='--')
                ax[0].plot(final_map_list[i][0, Ns * k : Ns * (k + 1),j],final_map_list[i][1, Ns * k : Ns * (k + 1),j], color='black')
                ax[1].plot(final_domain_list[i][0, Ns * k : Ns * (k + 1),j],final_domain_list[i][1, Ns * k : Ns * (k + 1),j], color='black')

        ax[0].grid('minor',  linestyle=':')
        ax[0].set_xlabel('x (-)')
        ax[0].set_ylabel('y (-)')
        ax[1].grid('minor',  linestyle=':')
        ax[1].set_xlabel('x (-)')
        ax[1].set_ylabel('y (-)')
        ax[0].set_title('mapped domain')
        ax[1].set_title('initial domain')
        ax[0].legend(("Nominal", "14th order", "ADS 14th order"),
          shadow=True, loc="best", handlelength=1.5, fontsize=16)
        fig.suptitle(' Time of analysis : '+ str(time_analysis[i]), fontsize=20)


# beginning of example:
def main():

    DA.init(1, 2) # initialize DA at lowest order possible, will need it later
    DA.setEps(1e-16)

    # example taken from https://doi.org/10.1007/s10569-015-9618-3
    XI = array.zeros(4)
    xb=0.008
    yb=0.08
    XI[0] += 1.0
    XI[3] += np.sqrt(1.5)

    TF = 50.
    T0 = 0.
    Ns = 33
    Ts = 51

    # part 1 of the example, assemble perimeter of ground truth domain:
    tgrid = np.linspace(T0, TF, Ts)

    xgrid = np.linspace(-1, 1, Ns)
    ygrid = np.linspace(-1, 1, Ns)

    lb = np.ones((Ns,2))
    lb[:, 0] = lb[:,0]*(-xb)
    lb[:, 1] = yb*ygrid

    rb = np.ones((Ns,2))
    rb[:, 0] = rb[:,0]*(xb)
    rb[:, 1] = yb*ygrid

    bb = np.ones((Ns,2))
    bb[:, 0] = xb*xgrid
    bb[:, 1] = bb[:,1]*(-yb)

    tb = np.ones((Ns,2))
    tb[:, 0] = xb*xgrid
    tb[:, 1] = tb[:,1]*(yb)

    perimeter = np.concatenate((tb, rb, bb, lb))
    perimeter_norm = np.concatenate((tb, rb, bb, lb))
    perimeter_norm[:, 0] = perimeter[:,0]/xb
    perimeter_norm[:, 1] = perimeter[:,1]/yb

    # propagation of ground truth perimeters (first run is going to take a while)
    try:
        # load the propagated domain if integration already exists
        with (thisfolder / 'ground_truth_propagations.npy').open('rb') as f:
            XF = np.load(f, allow_pickle = True)
    except FileNotFoundError:
        # integrate for half the orbital period
        XFtemp = np.zeros((4, Ts))
        XF = np.zeros((4,Ts,perimeter.shape[0]))

    # with DA.cache_manager():  # optional, for efficiency
        for j in range(perimeter.shape[0]):
            x0 = XI.copy()
            x0[0:2]+=perimeter[j,:]
            XFtemp[:,0]=x0.cons()
            for i in range(Ts-1):
                t0=tgrid[i]
                tf=tgrid[i+1]
                xf = RK78(x0, t0, tf, TBP)
                x0 = xf
                XFtemp[:,i+1]=xf.cons()
                print(tf)
            XF[:,:,j]=XFtemp

        with (thisfolder / 'ground_truth_propagations.npy').open('wb') as f:
            np.save(f, XF, allow_pickle = True)

    # compute only highest order propagation and lower orders evaluated by truncating poly map
    try:
        # load the propagated domain if integration already exists
        with (thisfolder / 'order_1.npy').open('rb') as f:
            XF1 = np.load(f, allow_pickle = True)
        with (thisfolder / 'order_2.npy').open('rb') as f:
            XF2 = np.load(f, allow_pickle = True)
        with (thisfolder / 'order_6.npy').open('rb') as f:
            XF6 = np.load(f, allow_pickle = True)
        with (thisfolder / 'order_14.npy').open('rb') as f:
            XF14 = np.load(f, allow_pickle = True)
    except FileNotFoundError:
        DA.init(14, 2)
        DA.setEps(1e-16)

        XI = array.zeros(4)
        XI[0] += 1.0 + xb * DA(1)
        XI[1] += 0.0 + yb * DA(2)
        XI[3] += np.sqrt(1.5)

        XFN = array.zeros((4,Ts))
        XFN[:, 0] = XI.copy()

    # with DA.cache_manager():  # optional, for efficiency
        x0 = array(XI)
        xf = array(XI)
        for i in range(Ts-1):
            t0 = tgrid[i]
            tf = tgrid[i+1]
            xf = RK78(x0, t0, tf, TBP)
            XFN[:, i+1] = xf.copy()

            x0 = xf
            print(tf)

        DA.pushTO(1)
        XF1 = np.zeros((4,Ts,perimeter_norm.shape[0]))
        x_sub = array.identity(2)
        for i in range(Ts):
            xf = XFN[:,i].copy()
            xf_temp = xf.eval(x_sub)
            for j in range(perimeter_norm.shape[0]):
                XF1[:, i, j] = xf_temp.eval(perimeter_norm[j, :])
        DA.popTO()

        DA.pushTO(2)
        XF2 = np.zeros((4,Ts,perimeter_norm.shape[0]))
        x_sub = array.identity(2)
        for i in range(Ts):
            xf=XFN[:,i].copy()
            xf_temp = xf.eval(x_sub)
            for j in range(perimeter_norm.shape[0]):
                XF2[:,i,j] = xf_temp.eval(perimeter_norm[j,:])
        DA.popTO()

        DA.pushTO(6)
        XF6 = np.zeros((4,Ts,perimeter_norm.shape[0]))
        x_sub = array.identity(2)
        for i in range(Ts):
            xf=XFN[:,i].copy()
            xf_temp=xf.eval(x_sub)
            for j in range(perimeter_norm.shape[0]):
                XF6[:,i,j]=xf_temp.eval(perimeter_norm[j,:])
        DA.popTO()

        XF14 = np.zeros((4,Ts,perimeter_norm.shape[0]))
        for i in range(Ts):
            xf=XFN[:,i].copy()
            for j in range(perimeter_norm.shape[0]):
                XF14[:,i,j]=xf.eval(perimeter_norm[j,:])

        with (thisfolder / 'order_1.npy').open('wb') as f:
            np.save(f, XF1, allow_pickle = True)
        with (thisfolder / 'order_2.npy').open('wb') as f:
            np.save(f, XF2, allow_pickle = True)
        with (thisfolder / 'order_6.npy').open('wb') as f:
            np.save(f, XF6, allow_pickle = True)
        with (thisfolder / 'order_14.npy').open('wb') as f:
            np.save(f, XF14, allow_pickle = True)


    ################ replication of first figures from paper
    ax1 = figure_1(XF, Ns, tgrid)
    ax2_1 = figure_2(XF, XF1, XF2, XF6, Ns, 16)
    ax2_2 = figure_2(XF, XF1, XF2, XF6, Ns, 33)
    ax2_3 = figure_2(XF, XF1, XF2, XF6, Ns, 34)
    ax3_2 = figure_3(XF, XF6, XF14, Ns, 33)
    ax3_3 = figure_3(XF, XF6, XF14, Ns, 34)
    ax3_4 = figure_3(XF, XF6, XF14, Ns, 35)
    ax3_6 = figure_3(XF, XF6, XF14, Ns, 38)
    ax3_8 = figure_3(XF, XF6, XF14, Ns, 39)


    ################################## Naive ADS application #######################
    # checks need to be performed at a later step
    DA.init(14, 2)
    DA.setEps(1e-16)
    XI = array.zeros(4)
    XI[0] += 1.0 + xb * DA(1)
    XI[1] += 0.0 + yb * DA(2)
    XI[3] += np.sqrt(1.5)
    init_domain = ADS(XI, [])

    init_list=[init_domain]
    final_lists=[]

    toll=1e-4
    Nmax=100

    time_analysis = [16, 33, 34, 35, 36, 37, 38, 39, 40]

    start_basic = time.time()
    for i in range(len(time_analysis)):
        final_list = ADS.eval(
            init_list, toll, Nmax,
            lambda domain: base_propagation(domain, t0=0, tf=time_analysis[i]))
        final_lists.append(final_list)

        print('time ', time_analysis[i], ' reached!')
    print('execution time base ADS: ', time.time() - start_basic)


    ax4 = figure_4(final_lists, XF, XF14, Ns, perimeter_norm, time_analysis)


    ####################### Advanced ADS application #######################

    init_domain = ADS(XI, [])

    init_list = [init_domain]
    final_lists = []
    final_list = init_list.copy()
    final_lists.append(final_list) # add also initial domains

    start_advanced = time.time()
    for i in range(Ts - 1):
        final_list = ADS.eval(
            final_list, toll, Nmax,
            lambda domain: advanced_propagation(domain, t0=tgrid[i], tf=tgrid[i+1]))
        final_lists.append(final_list)

        print('time ', tgrid[i+1], 'reached!')
    print('execution time advanced ADS: ', time.time() - start_advanced)


    ###################### Replication of ADS figures ######################
    figure_5(Ts, tgrid, final_lists)
    figure_6(final_lists, XF, XF14, Ns, perimeter_norm, time_analysis)

    plt.show()

    print('End')


if __name__ == "__main__":
    main()

"""
# Differential Algebra Core Engine in Python - DACEyPy

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import NDArray

import daceypy

from ._PrettyType import PrettyType


class RKstate(metaclass=PrettyType):
    epsabs: float = 0.0
    "Absolute tolerance"
    epsrel: float = 0.0
    "Relative tolerance"
    h: float = 0.0
    "current step size"
    h0: float = 0.0
    "initial step size"
    maxh: float = 0.0
    "max Step size"
    minh: float  = 0.0
    "min Step size"
    globalerr: float = 0.0
    "integration global error"
    t: Union[float, daceypy.DA] = None
    "current time"

class RKCoeff(metaclass=PrettyType):
    _RK_stage: int
    _RK_order: float
    _alpha: NDArray[np.double]
    _gamma: NDArray[np.double]
    _beta: NDArray[np.double]
    _beta_star: NDArray[np.double]

    @property
    def RK_stage(self):
        return self._RK_stage
    @property
    def RK_order(self):
        return self._RK_order
    @property
    def alpha(self):
        return self._alpha
    @property
    def gamma(self):
        return self._gamma
    @property
    def beta(self):
        return self._beta
    @property
    def beta_star(self):
        return self._beta_star


class RK78_DP(RKCoeff, metaclass=PrettyType):
    def __init__(self):
        self._RK_stage: int = 13
        self._RK_order: float = 7.0
        self._alpha: NDArray[np.double] = np.asarray([#0
                        1.0/18.0,         #1
                        1.0/48.0, 1.0/16.0,           #2
                        1.0/32.0,      0.0,    3.0/32.0,            #3
                        5.0/16.0,      0.0, -75.0/64.0,                     75.0/64.0,              #4
                        3.0/80.0,      0.0,        0.0,                      3.0/16.0,               3.0/20.0,            #5
        29443841.0/614563906.0,      0.0,        0.0,        77736538.0/692538347.0,     -28693883.0/1125000000.0,     23124283.0/1800000000.0,              #6
        16016141.0/946692911.0,      0.0,        0.0,        61564180.0/158732637.0,       22789713.0/633445777.0,    545815736.0/2771057229.0,    -180193667.0/1043307555.0,          #7
        39632708.0/573591083.0,      0.0,        0.0,      -433636366.0/683701615.0,    -421739975.0/2616292301.0,     100302831.0/723423059.0,      790204164.0/839813087.0,       800635310.0/3783071287.0,           #8
        246121993.0/1340847787.0,      0.0,        0.0,  -37695042795.0/15268766246.0,    -309121744.0/1061227803.0,      -12992083.0/49076693.0,    6005943493.0/2108947869.0,       393006217.0/1396673457.0,    123872331.0/1001029789.0,   #9
    -1028468189.0/846180014.0,      0.0,        0.0,      8478235783.0/508512852.0,    1311729495.0/1432422823.0, -10304129995.0/1701304382.0,  -48777925059.0/3047939560.0,     15336726248.0/1032824649.0, -45442868181.0/3398467696.0,  3065993473.0/597172653.0,  #10
        185892177.0/718116043.0,      0.0,        0.0,     -3185094517.0/667107341.0,    -477755414.0/1098053517.0,    -703635378.0/230739211.0,    5731566787.0/1027545527.0,       5232866602.0/850066563.0,   -4093664535.0/808688257.0,  3962137247.0/1805957418.0,  65686358.0/487910083.0, #11
        403863854.0/491063109.0,      0.0,        0.0,     -5068492393.0/434740067.0,     -411421997.0/543043805.0,     652783627.0/914296604.0,    11173962825.0/925320556.0,    -13158990841.0/6184727034.0,  3936647629.0/1978049680.0,   -160528059.0/685178525.0,  248638103.0/1413531060.0, 0.0]) #12
        self._gamma: NDArray[np.double] = np.asarray([
            0.0,  1.0/18.0,    1.0/12.0,      1.0/8.0,      5.0/16.0,         3.0/8.0,       59.0/400.0,   93.0/200.0,     5490023248.0/9719169821.0,    13.0/20.0,   1201146811.0/1299019798.0,        1.0,   1.0])
        self._beta: NDArray[np.double] = np.asarray([
            14005451.0/335480064.0, 0.0, 0.0, 0.0, 0.0,  -59238493.0/1068277825.0,   181606767.0/758867731.0,  561292985.0/797845732.0,  -1041891430.0/1371343529.0,  760417239.0/1151165299.0,  118820643.0/751138087.0, -528747749.0/2220607170.0, 1.0/4.0])
        self._beta_star: NDArray[np.double] = np.asarray([
            13451932.0/455176623.0, 0.0, 0.0, 0.0, 0.0,  -808719846.0/976000145.0, 1757004468.0/5645159321.0,  656045339.0/265891186.0,  -3867574721.0/1518517206.0,   465885868.0/322736535.0,   53011238.0/667516719.0,                  2.0/45.0,     0.0])


class RK78(RKCoeff, metaclass=PrettyType):
    def __init__(self):
        self._RK_stage: int = 13
        self._RK_order: float = 7.0
        self._alpha: NDArray[np.double] = np.asarray([
                #0
        2.0/27.0,         #1
        1.0/36.0, 1.0/12.0,           #2
        1.0/24.0,      0.0,    1.0/8.0,            #3
        5.0/12.0,      0.0, -25.0/16.0,    25.0/16.0,              #4
        1.0/20.0,      0.0,        0.0,         0.25,           0.2,            #5
    -25.0/108.0,      0.0,        0.0,  125.0/108.0,    -65.0/27.0,  125.0/54.0,              #6
        31.0/300.0,      0.0,        0.0,          0.0,    61.0/225.0,    -2.0/9.0,    13.0/900.0,          #7
            2.0,      0.0,        0.0,    -53.0/6.0,    704.0/45.0,  -107.0/9.0,     67.0/90.0,       3.0,           #8
    -91.0/108.0,      0.0,        0.0,   23.0/108.0,  -976.0/135.0,  311.0/54.0,    -19.0/60.0,  17.0/6.0,  -1.0/12.0,          #9
    2383.0/4100.0,      0.0,        0.0, -341.0/164.0, 4496.0/1025.0, -301.0/82.0, 2133.0/4100.0, 45.0/82.0, 45.0/164.0, 18.0/41.0,           #10
        3.0/205.0,      0.0,        0.0,          0.0,           0.0,   -6.0/41.0,    -3.0/205.0, -3.0/41.0,   3.0/41.0,  6.0/41.0,        0.0,           #11
    -1777.0/4100.0,      0.0,        0.0, -341.0/164.0, 4496.0/1025.0, -289.0/82.0, 2193.0/4100.0, 51.0/82.0, 33.0/164.0, 12.0/41.0,        0.0,        1.0])
        self._gamma: NDArray[np.double] = np.asarray([
            0.0,  2.0/27.0,    1.0/9.0,      1.0/6.0,      5.0/12.0,         0.5,       5.0/6.0,   1.0/6.0,    2.0/3.0,   1.0/3.0,        1.0,        0.0,  1.0 ])
        self._beta: NDArray[np.double] = np.asarray([
        41.0/840.0,      0.0,        0.0,          0.0,           0.0,  34.0/105.0,      9.0/35.0,  9.0/35.0,  9.0/280.0, 9.0/280.0, 41.0/840.0,        0.0,   0.0 ])
        self._beta_star: NDArray[np.double] = np.asarray([
            0.0,       0.0,        0.0,          0.0,           0.0,  34.0/105.0,      9.0/35.0,  9.0/35.0,  9.0/280.0, 9.0/280.0,        0.0, 41.0/840.0, 41.0/840.0 ])


class RK54(RKCoeff, metaclass=PrettyType):
    def __init__(self):
        self._RK_stage: int = 7
        self._RK_order: float = 5.0
        self._alpha: NDArray[np.double] = np.asarray([
            #0
        1.0/12.0,             #1
        1.0/12.0,      1.0/4.0,           #2
    55.0/324.0,  -25.0/108.0,  50.0/81.0,                #3
    83.0/330.0,   -13.0/22.0,  61.0/66.0,       9.0/110.0,               #4
    -19.0/28.0,      9.0/4.0,    1.0/7.0,       -27.0/7.0,       22.0/7.0,           #5
    19.0/200.0,          0.0,    3.0/5.0,    -243.0/400.0,      33.0/40.0,   7.0/80.0])
        self._gamma: NDArray[np.double] = np.asarray([
            0.0,      2.0/9.0,    1.0/3.0,         5.0/9.0,     2.0/3.0,       1.0,      1.0])
        self._beta: NDArray[np.double] = np.asarray([
    431.0/5000.0,        0.0, 333.0/500.0, -7857.0/10000.0,  957.0/1000.0, 193.0/2000.0,   -1.0/50.0])
        self._beta_star: NDArray[np.double] = np.asarray([
    19.0/200.0,          0.0,    3.0/5.0,    -243.0/400.0,      33.0/40.0,    7.0/80.0,     0.0])


class RK45(RKCoeff, metaclass=PrettyType):
    def __init__(self):
        self._RK_stage: int = 6
        self._RK_order: float = 4.0
        self._alpha: NDArray[np.double] = np.asarray([
            #0
        1.0/4.0,               #1
        3.0/32.0,        9.0/32.0,               #2
    1932.0/2197.0,  -7200.0/2197.0,  7296.0/2197.0,                 #3
    439.0/216.0,            -8.0,   3680.0/513.0,    -845.0/4104.0,            #4
        -8.0/27.0,             2.0, -3544.0/2565.0,    1859.0/4104.0,  -11.0/40.0])
        self._gamma: NDArray[np.double] = np.asarray([
            0.0,         1.0/4.0,       3.0/8.0,     12.0/13.0,         1.0,         0.5])
        self._beta: NDArray[np.double] = np.asarray([
        25.0/216.0,             0.0,  1408.0/2565.0,    2197.0/4104.0,       -0.2,         0.0])
        self._beta_star: NDArray[np.double] = np.asarray([
        16.0/135.0,             0.0, 6656.0/12825.0,  28561.0/56430.0,  -9.0/50.0,    2.0/55.0])

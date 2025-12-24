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

from typing import Callable, Optional, Tuple, Type, Union, overload

import numpy as np
from numpy.typing import NDArray

import daceypy

from ._PrettyType import PrettyType
from .RK import RK78, RKCoeff, RKstate


class integrator(metaclass=PrettyType):
    """
    Modular integrator class for DA objects
    """

    # *************************************************************************
    # *     Constructors & Destructors
    # *************************************************************************

    def __init__(
        self,
        RKcoeff: Optional[RKCoeff] = RK78(),
        stateType: Optional[Type] = np.ndarray,
    ) -> None:
        """
        Initialize a propagator object.

        Args:
            RKcoeff:
                instance of the class RKCoeff that stores the coefficients
                of the Numerical Propagation Scheme

        See also:
            RK.RKCoeff
            DA.getMonomial
        """

        # declaration of protected instance variables
        self._runningX: Union[daceypy.array, NDArray[np.double]] = None
        "current state"
        self._backX: Union[daceypy.array, NDArray[np.double]] = None
        "backward state"
        self._outX: Union[daceypy.array, NDArray[np.double]] = None
        "final state"
        self._stateType: Type = stateType

        self._t0: float = None
        "initial time"
        self._tf: float = None
        "final time (if I build well the Integrator it could be also befor of t0)"
        self._DT: float = None
        "support variable to store tf-t0 and avoid multiple type checks "
        self._propDir: int = 0
        "propagation direction (forward =1, backward=-1)"

        self._backTime: float = None
        "backward time"
        self._runningTime: float = None
        "current time"
        self._tout: float = None
        "final time of propagation"

        self._backH: float = None
        "backward timestep "
        self._epstol: NDArray[np.double] = None
        "Step tolerance"
        self._err: float = None
        "Error step tolerance"

        self._checkStep: bool = None
        "check if propagation step is accepted or if stepsize needs adjusting"
        self._reachstime: bool = None
        "check if final time is reached"

        self._input: RKstate = RKstate()
        "support object to store current state of the propagator"

        self._RKcoeff: RKCoeff = RKcoeff
        "support object to store Butcher's tableau of the propagation method"

        # perform dynamic assignment to avoid continous type checks
        if stateType is np.ndarray:
            setattr(integrator, '_geterr', _geterrFloat)
            setattr(integrator, '_getepstol', _getepstolFloat)
            self._inputType = np
        elif stateType is daceypy.array:
            setattr(integrator, '_geterr', _geterrDA)
            setattr(integrator, '_getepstol', _getepstolDA)
            self._inputType = daceypy.array

    # *************************************************************************
    # *     User Overloadable Routines
    # *************************************************************************

    def _CallBack_getStepSize(self) -> float:
        """
        Provides a user defined scaling of the optimal integration step
        (default = 1). Can be overloaded by the user.

        Returns:
            a user defined scaling factor (default is 1)
        """
        return self._DT

    def _CallBack_CheckEvent(self) -> bool:
        """
        Check if integration should stop according to an event function
        (default is False).
        Can be overloaded by the user.

        Returns:
            True if the integration needs to be stopped
        """
        if (self._checkStep):
        #User could define some event ot check within their own derived class
            self._checkStep = False
        return False

    def _acceptStep(
        self,
        pn: Union[daceypy.array, NDArray[np.double]],
        tnew: float,
    ) -> None:
        """
        Updates the integration state if the tolerances are satisfied.
        Can be overloaded by the user to define a custom set of actions.

        Args:
            pn: propagated state (either numpy.ndarray or daceypy.array)
            tnew: integration time reached
        """
        if self._err <= 1.0:
            self._checkStep = True
            self._runningX = pn
            self._input.t = tnew
            self._runningTime = tnew

    @staticmethod
    def f(
        x: Union[daceypy.array, NDArray[np.double]],
        t: float,
    ) -> Union[daceypy.array, NDArray[np.double]]:
        """
        Computes the righ-hand-side of the ODE.
        Must be overloaded by the user.

        Args:
            x: state (either numpy.ndarray or daceypy.array)
            t: time (float)

        Returns:
            The time derivatives of the state x at time t
            (either as numpy.ndarray or daceypy.array)
        """
        raise NotImplementedError()

    # *************************************************************************
    # *     Support Routines
    # *************************************************************************

    def _ReachsFinalTime(self) -> bool:
        """
        Checks if integration has reached final time.

        Returns:
            True if the integration should stop.
        """
        self._reachstime = (abs(1.0 - self._input.t / self._tf) <= 2.2e-16)
        return self._reachstime

    def _Initialize(
        self,
        Initset: Union[daceypy.array, NDArray[np.double]],
        t1: float,
        t2: float,
    ) -> None:
        """
        Initializes the state and time of the propagator.

        Args:
            Initset: initial state (either numpy.ndarray or daceypy.array)
            t1: initial time of the propagation
            t2: final time of the propagation

        Raises:
            TypeError
        """
        self._runningX = Initset
        self._runningTime = t1
        self._input.t = t1
        if type(self._runningX) is not self._stateType:
            raise TypeError(
                "propagator was initialized to propagate "
                f"(\"{self._stateType}\") however, "
                f"(\"{type(self._runningX)}\") was given")

    def _computeStep(self) -> \
            Tuple[Union[daceypy.array, NDArray[np.double]], Union[daceypy.array, NDArray[np.double]]]:
        """
        Computes the state after one propagation step according to the
        selected numerical scheme.

        Returns:
            Two states propagated at different orders to check error for stepsize adaptation.
        """
        Y = self._inputType.zeros((self._runningX.size, self._RKcoeff.RK_stage))
        Y[:, 0] = self._runningX.copy()
        feval = self.f(Y[:, 0], self._input.t + self._input.h * self._RKcoeff.gamma[0])
        pn = self._runningX + self._input.h * self._RKcoeff.beta[0] * feval
        pndiff = (self._RKcoeff.beta_star[0] - self._RKcoeff.beta[0]) * feval

        ia = 0
        for i in range(1, self._RKcoeff.RK_stage):
            Yk = self._inputType.zeros(self._runningX.size)
            for j in range(i):
                tj = self._input.t + self._input.h * self._RKcoeff.gamma[j]
                Yk += self._RKcoeff.alpha[ia] * self.f(Y[:, j], tj)
                ia += 1

            Y[:, i] = self._runningX + self._input.h*Yk
            feval = self.f(Y[:, i], self._input.t + self._input.h*self._RKcoeff.gamma[i])
            pn += self._input.h*self._RKcoeff.beta[i] * feval
            pndiff += (self._RKcoeff.beta_star[i] - self._RKcoeff.beta[i]) * feval
        pndiff *= self._input.h

        return pn, pndiff

    @overload
    def _geterr(self, pndiff: Union[daceypy.array, NDArray[np.double]]) -> float:
        """
        Computes the propagation error for the given timestep.

        Args:
            pndiff:
                Variation of state caused by different propagation orders.

        Returns:
            The normalized error to determine stepsize adaptation
        """
        ...

    @overload
    def _getepstol(self, pn: Union[daceypy.array, NDArray[np.double]]) -> NDArray[np.double]:
        """
        Computes the error tolerance for each step.

        Args:
            pn: state propagated according to current timestep.

        Returns:
            A vector of tolerances for each element of the state.
        """
        ...

    def _adaptStepSize(self) -> None:
        """
        Adapt the stepsize according to step error.
        """

        if self._err == 0.0:
            self._input.h=4.0
        else:
            self._input.h *= min(4.0, max(0.1, 0.9 * pow(1.0 / self._err, 1.0 / (self._RKcoeff.RK_order + 1.0))))
        

        if abs(self._input.h) <= self._input.minh * 1.2:
            self._input.h /= 3.0
            print(" --- WARNING MINIMUM STEP SIZE REACHED.")

        if abs(self._input.h) > self._input.maxh:
            self._input.h = self._propDir*self._input.maxh

        dt = abs(self._tf - self._input.t)
        if dt < abs(self._input.h):
            self._input.h = self._propDir*dt

    # *************************************************************************
    # *     Functions that should be accessed outside class
    # *************************************************************************
    def loadTime(self, t1: float, t2:  float) -> None:
        """
        Loads initial and final propagation times inside an instance of integrator class.

        Args:
            t1: initial time
            t2: final time
        """
        if not isinstance(t1, float):
            if isinstance(t1, int):
                t1=float(t1)
            else:
                raise TypeError(f"t1 must be a float (\"{type(t1)}\") was given")
        if not isinstance(t2, float):
            if isinstance(t2,int):
                t2=float(t2)
            else:
                raise TypeError(f"t2 must be a float (\"{type(t2)}\") was given")

        self._t0 = t1
        self._tf = t2

        self._DT = self._tf - self._t0
        self._propDir = 1 if self._DT > 0 else -1

    def loadTol(self, tol1: float = 0.0, tol2: float = 0.0) -> None:
        """
        Loads absolute and relative tolerances of integrator.

        Args:
            tol1: absolute tolerance
            tol2: relative tolerance
        """
        self._input.epsabs = tol1
        self._input.epsrel = tol2 or 1e-10

    def loadStepSize(self, h0: float = 0.0, max: float = 0.0, min: float = 0.0) -> None:
        """
        Loads initial, maximum, and minimum stepsize of integrator.

        Args:
            h0: initial stepsize
            max: maximum stepsize
            min: minimum stepsize
        """
        self._input.h = h0
        self._input.maxh = max
        self._input.minh = min

        if self._input.h == 0.0:
            self._input.h = self._propDir * np.log10(abs(self._DT) * 0.5 + 1.0)
        else:
            self._input.h = self._propDir * abs(self._input.h)

        self._input.h0 = self._input.h

        if self._input.maxh == 0.0:
            self._input.maxh = 0.4 * abs(self._DT)
        if self._input.minh == 0.0:
            self._input.minh = self._input.maxh * 2.2e-16

    def propagate(self, Initset: Union[daceypy.array, NDArray[np.double]], t1: float, t2: float) -> Union[daceypy.array, NDArray[np.double]]:
        """
        Propagates state in time interval according to selected numerical scheme.

        Args:
            Initset: initial state
            t1: initial time
            t2: final time

        Returns:
            The final state of the integration
        """
        self._Initialize(Initset, t1, t2)
        self._checkStep = False

        while not self._ReachsFinalTime() and not self._CallBack_CheckEvent():
            # ask user if wants to use dt as step size or not, as default dt is equal to optimal h
            dt = self._CallBack_getStepSize()
            self._input.h = abs(self._input.h) * dt / abs(self._DT)

            self._backX = self._runningX
            self._backTime = self._input.t
            self._backH = self._input.h
            v1, v1diff = self._computeStep()
            self._epstol = self._getepstol(v1)
            self._err = self._geterr(v1diff)
            self._acceptStep(v1, self._input.t + self._input.h)
            self._adaptStepSize()

        self._tout = self._input.t
        self._outX = self._runningX
        return self._runningX


# *************************************************************************
# *     Functions specialized for float type
# *************************************************************************

def _geterrFloat(self, pndiff: NDArray[np.double]) -> float:
    """
    Computes the propagation error for the given timestep.

    Args:
        pndiff:
            Variation of state caused by different propagation orders.

    Returns:
        The normalized error to determine stepsize adaptation
    """

    E: float = 0.0
    size: int = self._runningX.size

    temp = pow(pndiff / self._epstol, 2.0)
    E = sum(temp)

    return np.sqrt(E/size)

def _getepstolFloat(self, pn: NDArray[np.double]) -> NDArray[np.double]:
    """
    Computes the error tolerance for each step.

    Args:
        pn: state propagated according to current timestep.

    Returns:
        A vector of tolerances for each element of the state.
    """

    vconst = self._runningX
    v1const = pn

    tol = np.maximum(abs(vconst), abs(v1const))

    tol = self._input.epsabs + tol*self._input.epsrel

    return tol

# *************************************************************************
# *     Functions specialized for DA type
# *************************************************************************

def _geterrDA(self, pndiff: daceypy.array) -> float:
    """
    Computes the propagation error for the given timestep.

    Args:
        pndiff:
            Variation of state caused by different propagation orders.

    Returns:
        The normalized error to determine stepsize adaptation
    """

    E: float = 0.0
    size: int = self._runningX.size

    temp = pndiff.cons()

    E = sum(pow(temp/self._epstol, 2.0))

    return np.sqrt(E/size)


def _getepstolDA(self, pn: daceypy.array) -> NDArray[np.double]:
    """
    Computes the error tolerance for each step.

    Args:
        pn: state propagated according to current timestep.

    Returns:
        A vector of tolerances for each element of the state.
    """

    vconst = self._runningX.cons()
    v1const = pn.cons()

    tol = np.maximum(abs(vconst), abs(v1const))

    tol = self._input.epsabs + tol*self._input.epsrel

    return tol


def PicardLindelof(
    x: Union[daceypy.array, NDArray[np.double]],
    direction: int, 
    tf: float,
    f: Callable[[daceypy.array, Union[daceypy.DA, float]], daceypy.array],
) -> daceypy.array:
    """
    Computes the time expansion at final time with Picard Lindelof operator

    Args:
        x: expansion state.
        direction: index of DA variable corresponding to time.
        tf: constant float expressing the reference time of expansion.
        f: right hand side of ODE.

    Returns:
        The input state expanded in time about input reference time.
    """
    expansionOrder = daceypy.DA.getMaxOrder()
    t = tf + daceypy.DA(direction)

    x_i = x.copy()

    for i in range(expansionOrder):
        # iteration n times
        x_i = x + f(x_i, t).integ(direction)

    return x_i

class integrator_optimized(integrator):
    """
    Extended integrator optimised for reduced computational time, supporting time-vector propagation and both stepwise and direct DA variants.
    """

    def __init__(
        self,
        RKcoeff: Optional[RKCoeff] = RK78(),
        stateType: Optional[Type] = np.ndarray,
        DA_type="DA_direct",
    ) -> None:

        # call parent constructor
        super().__init__(RKcoeff, stateType)

        # store additional attribute on DA type
        self.DA_type = DA_type

    def _Initialize(
        self,
        Initset: Union[daceypy.array, NDArray[np.double]],
        t0: float,
    ) -> None:
        """
        Initializes the state and time of the propagator.

        Args:
            Initset: initial state (either numpy.ndarray or daceypy.array)
            t0: initial time of the propagation
        Raises:
            TypeError
        """
        self._runningX = Initset
        self._runningTime = t0
        self._input.t = t0
        if type(self._runningX) is not self._stateType:
            raise TypeError(
                "propagator was initialized to propagate "
                f"(\"{self._stateType}\") however, "
                f"(\"{type(self._runningX)}\") was given")
    

    def _computeStep(self) -> \
        Tuple[Union[daceypy.array, NDArray[np.double]], Union[daceypy.array, NDArray[np.double]]]:
        """
        Computes the state after one propagation step according to the
        selected numerical scheme.
        Returns:
            Two states propagated at different orders to check error for stepsize adaptation.
        """
        # Preallocate
        n_states = self._runningX.size
        n_stages = self._RKcoeff.RK_stage
        Y = self._inputType.zeros((n_states, n_stages))
        F = self._inputType.zeros((n_states, n_stages))
        
        # Precompute all h-multiplied coefficients
        h = self._input.h
        t_base = self._input.t
        h_gamma = h * self._RKcoeff.gamma
        h_alpha = h * self._RKcoeff.alpha
        h_beta = h * self._RKcoeff.beta
        
        # Build alpha matrix: alpha_matrix[i, j] contains the coefficient for stage i, previous stage j
        alpha_matrix = np.zeros((n_stages, n_stages))
        ia = 0
        for i in range(1, n_stages):
            for j in range(i):
                alpha_matrix[i, j] = h_alpha[ia]
                ia += 1
        
        # Initial state
        Y[:, 0] = self._runningX
        F[:, 0] = self.f(Y[:, 0], t_base + h_gamma[0])
        
        # Initialize solution and error
        pn = self._runningX + h_beta[0] * F[:, 0]
        pndiff = (self._RKcoeff.beta_star[0] - self._RKcoeff.beta[0]) * F[:, 0]
        
        # Remaining stages (vectorized)
        for i in range(1, n_stages):
            # Vectorized: matrix-vector multiplication replaces inner loop
            # F[:, :i] is (n_states x i), alpha_matrix[i, :i] is (i,)
            # Result is (n_states,)
            Yk = F[:, :i] @ alpha_matrix[i, :i]
            
            Y[:, i] = self._runningX + Yk
            F[:, i] = self.f(Y[:, i], t_base + h_gamma[i])
            
            # Update solution and error estimate
            pn += h_beta[i] * F[:, i]
            pndiff += (self._RKcoeff.beta_star[i] - self._RKcoeff.beta[i]) * F[:, i]
        
        pndiff *= h
        
        return pn, pndiff

    def _adaptStepSize(self) -> None:
        """
        Adapt the stepsize according to step error.
        """
        if self._err == 0.0:
            self._input.h = 4.0
        else:
            # Compute scaling factor
            exponent = 1.0 / (self._RKcoeff.RK_order + 1.0)
            scale = 0.9 * pow(1.0 / self._err, exponent)
            scale = min(4.0, max(0.1, scale))
            self._input.h *= scale
        
        # Get absolute step size once
        abs_h = abs(self._input.h)
        
        # Check minimum step size
        if abs_h <= self._input.minh * 1.2:
            self._input.h /= 3.0
            print(" --- WARNING MINIMUM STEP SIZE REACHED.")
            abs_h /= 3.0  # Update abs_h to avoid recomputing
        
        # Enforce maximum step size (reuse abs_h)
        if abs_h > self._input.maxh:
            self._input.h = self._propDir * self._input.maxh
            abs_h = self._input.maxh  # Update for next check
        
        # Ensure we don't overshoot final time (reuse abs_h)
        dt_remaining = abs(self._tf - self._input.t)
        if dt_remaining < abs_h:
            self._input.h = self._propDir * dt_remaining

    def propagate(self, Initset: Union[daceypy.array, NDArray[np.double]], *args):
        """
        Propagates the state and returns the state evaluated at all times in t_eval.
        Supports both forward (t0 < tf) and backward (t0 > tf) propagation.
        
        DA_direct: Forces integration steps exactly on t_eval points (NO interpolation)
        DA_stepwise: Forces integration steps exactly on t_eval points, reinitializes DA at each point
        """
        t_eval = self._parse_time_arguments(*args)
        self._initialize_propagation(Initset, t_eval)
        
        # Determine propagation direction
        forward = t_eval[-1] >= t_eval[0]
        
        states = []
        t_eval_index = 0
        
        # Store initial condition if requested
        if t_eval_index < len(t_eval) and abs(t_eval[t_eval_index] - self._input.t) < 1e-12:
            states.append(self._runningX.copy())
            t_eval_index += 1
        
        # Prepare DA identity for stepwise mode
        if self.DA_type == "DA_stepwise":
            dim_state = len(self._runningX)
            da_id = daceypy.array.identity(dim_state)
        
        # ---------- Integration loop ----------
        while not self._ReachsFinalTime() and not self._CallBack_CheckEvent():
            # Check if all eval points processed
            if t_eval_index >= len(t_eval):
                break
            
            next_teval = t_eval[t_eval_index]
            
            # Prepare and adjust step
            self._prepare_integration_step()
            
            # Adjust step size to land exactly on next t_eval point
            would_overshoot = ((self._input.t + self._input.h > next_teval) if forward 
                            else (self._input.t + self._input.h < next_teval))
            
            if would_overshoot:
                self._input.h = next_teval - self._input.t
            
            # Execute and finalize step
            v1 = self._execute_integration_step()
            self._finalize_integration_step(v1)
            
            # Store state if we landed on a t_eval point
            if self._input.t == next_teval:
                states.append(self._runningX.copy())
                t_eval_index += 1
                
                # Reinitialize DA expansion for stepwise mode
                if self.DA_type == "DA_stepwise" and t_eval_index < len(t_eval):
                    const_part = self._runningX.cons()
                    self._runningX = da_id.copy()
                    self._runningX += const_part
        
        self._set_final_outputs()
        return states


    def _parse_time_arguments(self, *args):
        """Parse and validate time arguments. Does NOT sort - preserves direction."""
        if len(args) == 2:  # (t0, tf)
            t0, tf = args
            t_eval = np.array([t0, tf], dtype=float)
        elif len(args) == 1:  # (t_eval,)
            t_eval = args[0]
            if np.isscalar(t_eval):
                t_eval = np.array([0.0, float(t_eval)])
            else:
                t_eval = np.asarray(t_eval, dtype=float)
        else:
            raise TypeError("integrate() accepts either (t0, tf) or (t_eval,)")
        
        return t_eval


    def _initialize_propagation(self, Initset: Union[daceypy.array, NDArray[np.double]], t_eval:np.ndarray):
        """Initialize the integrator for propagation."""
        self._Initialize(Initset, t_eval[0])
        self._checkStep = False


    def _prepare_integration_step(self):
        """Prepare for the next integration step."""
        dt = self._CallBack_getStepSize()
        self._input.h = abs(self._input.h) * dt / abs(self._DT)
        
        # Backup state
        self._backX = self._runningX
        self._backTime = self._input.t
        self._backH = self._input.h


    def _execute_integration_step(self):
        """Execute one integration step and return results."""
        v1, v1diff = self._computeStep()
        self._epstol = self._getepstol(v1)
        self._err = self._geterr(v1diff)
        return v1


    def _finalize_integration_step(self, v1:  Union[daceypy.array, NDArray[np.double]]):
        """Accept step and adapt step size."""
        t_new = self._input.t + self._input.h
        self._acceptStep(v1, t_new)
        self._adaptStepSize()


    def _set_final_outputs(self):
        """Set final output values."""
        self._tout = self._input.t
        self._outX = self._runningX







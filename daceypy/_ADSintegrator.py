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

from typing import Tuple, Union, Optional, List

import numpy as np
from numpy.typing import NDArray

import daceypy

from .RK import RKCoeff, RK78

from ._integrator import integrator, integrator_optimized

from ._PrettyType import PrettyType

from ._ADS import ADS


class ADSstate(metaclass=PrettyType):
    """
    online ADS (Adaptive Domain Splitting) base element.
    """
    def __init__(
        self,
        ADSPatch: ADS,
        time: float = 0.0,
        stepsize: float = 0.0,
        checkBreached: bool = False,
        splitTimes: List[float] = [],
    ):
        """
        Initialize an instance of ADSstate class for propagation.

        Args:
            ADSPatch:
              instance of the class ADS that stores the current patch
              under analysis
            time: reference epoch for the patch
            stepsize: stepsize of the integrator at the patch epoch
            checkBreached: 
              check if patch needed split even though it was impossible
              during its propagation
            splitTimes: list of split times for the current patch

        See also:
            ADS
        """
        # ADSstate if a support class, its instances are used to store 
        # ADS info during propagation.

        self.ADSPatch: ADS = ADSPatch
        "The current ADS patch"
        self.time: float = time
        "Time at witch current patch started existing"
        self.stepsize: float = stepsize
        "Stepsize of the integrator when this patch was created"
        
        # moreover it has additional support variables to store ADS 
        # info in a easily accessible way.

        self.splitTimes: List[float] = splitTimes
        "List of split times for each patch"
        self.checkBreached: bool = checkBreached
        "Check if domain can be further split"
        # defined as "protected" because it should only be modified by 
        # a couple of class methods 
        self._breachTime: float = -1.0
        "Time at which split became necessary even though impossible"


class ADSintegrator(integrator, metaclass=PrettyType):
    """
    ADS integrator to perform splits online during propagation.
    """
    def __init__(self, RKcoeff: RKCoeff = RK78()) -> None:
        """
        Initialize an instance of the custom propagator capable of 
        dealing with online ADS.

        Args:
            RKcoeff:
              instance of the class RKCoeff that stores the coefficients
              of the Numerical Propagation Scheme

        See also:
            integrator
            RK.RKCoeff
        """

        # call to parent class constructor:
        # ADS is only available for daceypy.array type
        super(ADSintegrator, self).__init__(RKcoeff, daceypy.array)

        self._nSplitMax: int = 0
        "Maximum number of split for the ADS routine"

        self._errtol: Optional[Union[float, NDArray[np.double]]] = None
        "maximum truncation tolerance to determine split condition"
        
        self._stack: Optional[ADSstate] = None
        "ADS patch at the top of the stack with enhanced info"

    def _InitializeList(
        self,
        set: List[ADS],
        splitTimesList: Optional[List[List[float]]] = None,
    ) -> List[ADSstate]:
        """
        Initializes the list of ADS objects that need propagation as
        instances of ADSstate required by this custom propagator.

        Args:
            set: initial list of ADS objects.
            splitTimesList:
              optional list of split times of each element of set. 
              This is only required if one wants to propagate a domain
              that was already split while preserving its history.

        Returns:
            List of ADSstate instances ready for online ADS propagation

        Raises:
            ValueError
        """

        # NB: needs to be called after integrator.loadTime and 
        # integrator.loadStepSize
 
        out: List[ADSstate] = []

        # this part is only necessary if a pre-split domain is provided
        # and one wants to retain the split history (times).
        # If a list of split times is provided it must have proper size.
        # If it is not provided the info will be lost and only the new 
        # splits will be saved at the end of the propagation.
        if not splitTimesList:
            splitTimesList = [[]] * len(set)
        elif len(splitTimesList) != len(set):
            raise ValueError(
                "specified splitTimesList must have the same length"
                "as the number of elements in the initial set.")
        else:
            for i in range(len(set)):
                if len(set[i].nsplit) != len(splitTimesList[i]):
                    raise ValueError(
                        f"specified splitTimesList[{i}] must have the "
                        "same length as the number of splits in the "
                        "corresponding set element.")
    
        # create a list of ADSstate instances from the initial domains
        for i in range(len(set)):
            temp = ADSstate(
                set[i],
                self._t0,
                self._input.h,
                splitTimes = splitTimesList[i],
            )

            out.append(temp)

        # element at the top of the stack is stored in a variable
        self._stack = temp
    
        return out
    
    def _CallBack_CheckEvent(self) -> bool:
        """
        Check if integration should stop according to an event function
        (default is False). This version overloads that of integrator.

        Returns:
            True if the integration needs to be stopped

        See also:
            integrator._CallBack_CheckEvent
        """
        # function that overloads integrator.CallBack_CheckEvent:
        # If the current domain has already been tested and seen that 
        # cannot be split simply go ahead without doing anything
        if self._stack.checkBreached:
            self._checkStep = False
        else:
            if self._checkStep:
                # if the integration step was accepted check for split
                temp_f = ADS(
                    self._stack.ADSPatch.box,
                    self._stack.ADSPatch.nsplit,
                    self._runningX,
                )
                
                # if the domain can be split
                if temp_f.canSplit(self._nSplitMax): 
                    # If step is accepted and needs to be checked 
                    # but a violation is detected simply roll back 
                    # to previous time step
                    # If step is not accepted will not be checked
                    # and hence execution will never arrive here
                    if temp_f.checkSplit(self._errtol):
                        self._runningX = self._backX
                        self._input.t = self._backTime
                        self._input.h = self._backH
                    else:
                        self._checkStep = False
                else:
                    # the domain cannot be further split. 
                    # Set time of violation and violation flag 
                    # so that check can be avoided until tf 
                    self._stack.checkBreached = True
                    self._stack._breachTime = self._input.t
                    self._checkStep = False

        return self._checkStep
    
    def _SplitStateProcess(
        self,
        xf: daceypy.array,
        listIn: List[ADSstate],
        listOut: List[ADSstate],
    ) -> Tuple[List[ADSstate], List[ADSstate]]:
        """
        Updates the list of ADS objects that need propagation online.

        Args:
            xf: current state of the propagation.
            listIn: list of ADSstates that still need to be propagated.
            listOut: list of ADSstates that reached tf.

        Returns:
            List of ADSstate instances that still need propagation
            List of ADSstate instances that reached tf
            
        See also:
            ADS
        """
        # create temporary ads patch with latest propagator state
        state = ADS(
            self._stack.ADSPatch.box,
            self._stack.ADSPatch.nsplit,
            xf,
        )

        # if not at end of propagation this means that a split was 
        # necessary (and possible) from how we defined the checkevent
        if not self._reachstime:

            print(
                "The domain was split at instant ",
                self._input.t,
                " and continued the propagation...")
            
            # compute split subdomains
            dir = state.direction()
            Dl, Dr = state.split(dir)
            
            # update list of split times for each subdomain
            tempSplitTimes = self._stack.splitTimes.copy()
            tempSplitTimes.append(self._input.t)

            # add elements to the list of objects that need propagation
            tempL = ADSstate(
                Dl,
                self._input.t,
                self._input.h,
                splitTimes = tempSplitTimes,
            )
            tempR = ADSstate(
                Dr,
                self._input.t,
                self._input.h,
                splitTimes = tempSplitTimes,
            )
            
            # add one of the domains at the top of the stack
            self._stack = tempR
            # add other domain to list of element that need propagation
            listIn.append(tempL)
        else:
            # only here if reached tf (regardless of accuracy breach)
            tempD = ADSstate(
                state,
                self._input.t,
                self._input.h,
                self._stack.checkBreached,
                splitTimes = self._stack.splitTimes,
            )
            
            tempD._breachTime = self._stack._breachTime
            
            # add domain to final list: this step could be moved
            # outside of this function to avoid continuously passing 
            # listOut in and out of this function. It is only kept here
            # to simplify the propagate function but this could be
            # changed in the future for efficiency by simply passing
            # tempD as output.

            listOut.append(tempD)

        return listIn, listOut


    def _importfromList(self, ListIn: List[ADSstate]) -> None:
        """
        Remove from the list of ADS elements that need propagation the 
        one at the top of the stack and prepare it for propagation.

        Args:
            listIn: list of ADSstates that still need to be propagated.
            
        See also:
            integrator
        """
        # put first element to top of the stack
        self._stack = ListIn.pop()
        # pass its time and stepsize to the propagator
        self._input.t = self._stack.time
        self._input.h = self._stack.stepsize
        # set condition that it has not reached final time
        self._ReachsFinalTime() 

    def loadADSopt(
        self,
        tol: Union[float, NDArray[np.double]] = 1e-4,
        nsplit: int = 15,
    ) -> None:
        """
        Load ADS options.

        Args:
            tol: maximum truncation error tolerance.
            nsplit: maximum number of splits per each domain.
        """

        self._errtol = tol
        self._nSplitMax = nsplit

    def propagate(
        self,
        set: List[ADS],
        t1: float,
        t2: float,
        splitTimesList: List[List[float]] = [],
    ) -> List[ADSstate]:
        """
        Propagates the initial list of ADS objects while dealing with 
        online ADS. Overloads parent class method integrator.propagate.

        Args:
            List: list of ADS elements that need propagation.
            t1: initial reference time for each element of the List.
            t2: final time of the propagation.
            splitTimesList:
              optional list of split times of each element of set. 
              This is only required if one wants to propagate a domain
              that was already split while preserving its history.

        Returns:
            List of ADSstate instances that reached tf
            
        See also:
            integrator.propagate

        NB: t1 and t2 are not used in this function. They are only left
        here to maintain same syntax as the parent class method that 
        this function overloads. The initial time of the propagation 
        MUST be set through integrator.loadTime and it will be adapted
        online by each patch. Conversely, tf is fixed and it MUST be
        set through integrator.loadTime.
        """
        
        # list of final domains
        listOut: List[ADSstate] = []
        # create initial list from initial set of domains
        listIn = self._InitializeList(set, splitTimesList)

        print("Let's start propagation...")
        # loop until list of domains is not empty
        while listIn:
            # remove first domain and put it at the top of the stack
            self._importfromList(listIn)
            # loop until final time is reached for the domain
            while not self._reachstime:
                xf = super(ADSintegrator, self).propagate(self._stack.ADSPatch.manifold, self._input.t, t2)
                # after each call to propagation update list of patches
                listIn, listOut = self._SplitStateProcess(xf, listIn, listOut)

        print("Propagation completed. ")

        return listOut

    

class ADSintegrator_optimized(integrator_optimized, metaclass=PrettyType):
    """
    ADS integrator to perform splits online during propagation.
    """
    def __init__(self, RKcoeff: RKCoeff = RK78()) -> None:
        """
        Initialize an instance of the custom propagator capable of 
        dealing with online ADS.

        Args:
            RKcoeff:
              instance of the class RKCoeff that stores the coefficients
              of the Numerical Propagation Scheme

        See also:
            integrator
            RK.RKCoeff
        """

        # call to parent class constructor:
        # ADS is only available for daceypy.array type
        super(ADSintegrator_optimized, self).__init__(RKcoeff, daceypy.array)

        self._nSplitMax: int = 0
        "Maximum number of split for the ADS routine"

        self._errtol: Optional[Union[float, NDArray[np.double]]] = None
        "maximum truncation tolerance to determine split condition"
        
        self._stack: Optional[ADSstate] = None
        "ADS patch at the top of the stack with enhanced info"

    def _InitializeList(
        self,
        set: List[ADS],
        splitTimesList: Optional[List[List[float]]] = None,
    ) -> List[ADSstate]:
        """
        Initializes the list of ADS objects that need propagation as
        instances of ADSstate required by this custom propagator.

        Args:
            set: initial list of ADS objects.
            splitTimesList:
              optional list of split times of each element of set. 
              This is only required if one wants to propagate a domain
              that was already split while preserving its history.

        Returns:
            List of ADSstate instances ready for online ADS propagation

        Raises:
            ValueError
        """

        return ADSintegrator._InitializeList(self, set, splitTimesList)
    
    def _CallBack_CheckEvent(self) -> bool:
        """
        Check if integration should stop according to an event function
        (default is False). This version overloads that of integrator.

        Returns:
            True if the integration needs to be stopped

        See also:
            integrator._CallBack_CheckEvent
        """
        return ADSintegrator._CallBack_CheckEvent(self)
    
    def _SplitStateProcess_optimized(
        self,
        xf: daceypy.array,
        listIn: List[ADSstate],
        t_vect: np.ndarray,
    ) -> Tuple[List[ADSstate], List[ADSstate]]:
        """
        Updates the list of ADS objects that need propagation online.
        
        Args:
            xf: current state of the propagation.
            listIn: list of ADSstates that still need to be propagated.
            listOut: list of ADSstates that reached tf.
            
        Returns:
            List of ADSstate instances that still need propagation
            List of ADSstate instances that reached tf
            
        See also:
            ADS
        """
        # Initialize output list for ADSstates that completed propagation
        listOut = []
        
        # Check if propagation stopped before reaching final time
        # (indicates split event was triggered)
        if not self._reachstime:
            # Process intermediate propagation states if they exist
            if len(t_vect) > 0:
                # Store all intermediate states (all except the last one)
                # These represent the propagation trajectory before the split
                for i in range(len(xf) - 1):
                    # Create ADSstate for each intermediate point
                    tempD = ADSstate(
                        ADS(
                            self._stack.ADSPatch.box,  # Affine DA map of the domain's parallelepiped
                            self._stack.ADSPatch.nsplit,  # Split order parameter
                            xf[i],  # State at this time step
                        ),
                        t_vect[i],  # Time corresponding to this state
                        None,  # No step size (intermediate state)
                        self._stack.checkBreached,  # Accuracy breach check function
                        splitTimes=self._stack.splitTimes.copy(),  # History of split times
                    )
                    # Preserve breach time information from stack
                    tempD._breachTime = self._stack._breachTime
                    # Add to output list (completed states)
                    listOut.append(tempD)
                
            # Create ADS object from the final state where split occurs
            stateIn = ADS(
                self._stack.ADSPatch.box,
                self._stack.ADSPatch.nsplit,
                xf[-1],  # Last propagated state before split
            )
            
            print(
                "The domain was split at instant ",
                self._input.t,
                " and continued the propagation..."
            )
            
            # Choose the split direction with the greedy heuristic in
            # direction(), based on truncation error, not domain expansion
            dir = stateIn.direction()
            # Split the domain into left and right subdomains
            Dl, Dr = stateIn.split(dir)
            
            # Update split times history by adding current split time
            tempSplitTimes = self._stack.splitTimes.copy()
            tempSplitTimes.append(self._input.t)
            
            # Create ADSstate for left subdomain
            tempL = ADSstate(
                Dl,  # Left subdomain
                self._input.t,  # Current time (split time)
                self._input.h,  # Time step for next propagation
                splitTimes=tempSplitTimes,  # Updated split history
            )
            
            # Create ADSstate for right subdomain
            tempR = ADSstate(
                Dr,  # Right subdomain
                self._input.t,  # Current time (split time)
                self._input.h,  # Time step for next propagation
                splitTimes=tempSplitTimes,  # Updated split history
            )
            
            # Place right subdomain on stack for immediate processing
            self._stack = tempR
            # Add left subdomain to queue for later propagation
            listIn.append(tempL)
        else:
            # Propagation reached final time tf successfully
            # Store all final states (no split needed)
            for i in range(len(xf)):
                # Create ADSstate for each final state
                tempD = ADSstate(
                    ADS(
                        self._stack.ADSPatch.box,
                        self._stack.ADSPatch.nsplit,
                        xf[i],  # Final propagated state
                    ),
                    t_vect[i],  # Corresponding time
                    None,  # No further propagation needed
                    self._stack.checkBreached,
                    splitTimes=self._stack.splitTimes.copy(),
                )
                # Preserve breach time information
                tempD._breachTime = self._stack._breachTime
                # Add to output list (completed propagation)
                listOut.append(tempD)
            pass
        
        # Return updated lists:
        # - listIn: domains that still need propagation (including new splits)
        # - listOut: domains that completed propagation to tf
        return listIn, listOut

    def _importfromList(self, ListIn: List[ADSstate]) -> None:
        """
        Remove from the list of ADS elements that need propagation the 
        one at the top of the stack and prepare it for propagation.

        Args:
            listIn: list of ADSstates that still need to be propagated.
            
        See also:
            integrator
        """
        ADSintegrator._importfromList(self, ListIn)

    def loadADSopt(
        self,
        tol: Union[float, NDArray[np.double]] = 1e-4,
        nsplit: int = 15,
    ) -> None:
        """
        Load ADS options.

        Args:
            tol: maximum truncation error tolerance.
            nsplit: maximum number of splits per each domain.
        """
        ADSintegrator.loadADSopt(self, tol, nsplit)

    def propagate(
        self,
        set: List[ADS],
        t_vect: np.ndarray,
        splitTimesList: List[List[float]] = None,
    ) -> List[List[ADSstate]]:
        """
        Propagates the initial list of ADS objects while dealing with 
        online ADS. Overloads parent class method integrator.propagate.
        
        Args:
            set: List of ADS elements that need propagation.
            t_vect: Array of time points at which to output propagated states.
            splitTimesList: Optional list of split times for each element of set. 
                Required only when propagating a domain that was already split 
                while preserving its history.
        
        Returns:
            List of lists of ADSstate instances, one list per time point in t_vect.
        
        See also:
            integrator.propagate
        
        Note:
            The initial time of the propagation MUST be set through
            integrator.loadTime and will be adapted online by each patch.
            The final time tf is fixed and MUST be set through integrator.loadTime.

        Raises:
            NotImplementedError: if loadTime was configured for backward
                propagation (t0 > tf). The time-range masks used to track
                requested output points and split legs assume forward
                propagation; used backward they select the wrong points
                without any error, so backward propagation is rejected
                explicitly here instead.
            ValueError: if t_vect does not reach self._tf. Each domain is
                propagated leg by leg towards the next requested point in
                t_vect; once every point has been consumed the loop needs
                a further target to keep advancing to self._tf, which is
                only available if t_vect itself extends that far.
        """
        if self._propDir < 0:
            raise NotImplementedError(
                "ADSintegrator_optimized.propagate only supports forward "
                "propagation (t0 < tf)."
            )
        if len(t_vect) == 0 or t_vect[-1] < self._tf:
            raise ValueError(
                "t_vect must include a final point at (or beyond) tf "
                f"(={self._tf}); got t_vect ending at "
                f"{t_vect[-1] if len(t_vect) else 'empty'}."
            )

        # Create initial processing queue from input domains
        listIn = self._InitializeList(set, splitTimesList)
        
        print("Let's start propagation...")
        
        # Initialize output structure: one list of states per time point
        listOut_tot: List[List[ADSstate]] = [[] for _ in range(len(t_vect))]
        
        # Process all domains in the queue
        while listIn:
            # Extract and load the next domain from the queue
            self._importfromList(listIn)
            last_t = self._input.t
            
            # Propagate the current domain until final time is reached
            while not self._reachstime:
                # Construct time vector: current time + all future requested times
                t_vect_in = np.concatenate(
                    ([self._input.t], t_vect[t_vect > self._input.t])
                )

                # Perform numerical integration step
                xf = super(ADSintegrator_optimized, self).propagate(
                    self._stack.ADSPatch.manifold,
                    t_vect_in
                )

                # Remove intermediate state if current time was not in original t_vect
                if t_vect_in[0] not in t_vect and len(xf) > 1:
                    xf = xf[1:]

                # Identify time points covered in this propagation step
                mask_prop = (t_vect <= self._input.t) & (t_vect >= last_t)
                indices_prop = np.where(mask_prop)[0]
                t_vect_prop = t_vect[indices_prop]
                
                # Handle domain splitting and update processing queues
                listIn, listOut = self._SplitStateProcess_optimized(
                    xf, listIn, t_vect_prop
                )
                
                # Store output states at their corresponding time indices
                if len(listOut) > 0 and len(indices_prop) > 0:
                    # Determine which time indices to store based on propagation status
                    if not self._reachstime and len(t_vect_prop) > 0:
                        # Mid-propagation: exclude final state if it matches current time
                        # (will be reprocessed in next iteration for accuracy)
                        if t_vect_prop[-1] == self._input.t:
                            indices_to_use = indices_prop[:-1]
                        else:
                            indices_to_use = indices_prop
                    else:
                        # Final time reached: include all states
                        indices_to_use = indices_prop

                    # Append computed states to output structure
                    for i, idx in enumerate(indices_to_use):
                        listOut_tot[idx].append(listOut[i])
                
                # Update time tracker for next iteration
                last_t = self._input.t
        
        print("Propagation completed.")
        return listOut_tot

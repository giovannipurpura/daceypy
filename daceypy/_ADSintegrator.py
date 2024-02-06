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

from typing import (Tuple, Union, Optional, List, Type)

import numpy as np
from numpy.typing import NDArray

import daceypy

from .RK import RKCoeff, RK78

from ._integrator import integrator

from ._PrettyType import PrettyType

from ._ADS import ADS


class ADSstate(metaclass=PrettyType):
    """
    online ADS (Adaptive Domain Splitting) base element.
    """
    def __init__(
        self, 
        ADSPatch: ADS = None, 
        time: float = 0.0, 
        stepsize : float = 0.0, 
        checkBreached: Optional[bool] = False, 
        splitTimes: Optional[List[float]] = [],
        ):
        """
        Initialize an instance of ADSstate class for propagation.

        Args:
            ADSParch:
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

        self.ADSPatch : ADS = ADSPatch
        "The current ADS patch"
        self.time: float = time
        "Time at witch current patch started existing"
        self.stepsize: float = stepsize
        "Stepsize of the integrator when this patch was created"
        
        # moreover it has additional support variables to store ADS 
        # info in a easily accessible way.

        self.splitTimes: list[float] = splitTimes
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
    def __init__(self, RKcoeff: Optional[RKCoeff]=RK78()) -> None:
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

              
        self._errtol: Union[float, NDArray[np.double]] = None
        "maximum truncation tolerance to determine split condition"
        
        self._stack: ADSstate = None 
        "ADS patch at the top of the stack with enhanced info"
        
        return

    def _InitializeList(
        self, 
        set: List[ADS], 
        splitTimesList: Optional(List[List[float]]) = [],
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
 
        out : List[ADSstate] = []

        # this part is only necessary if a pre-split domain is provided
        # and one wants to retain the split history (times).
        # If a list of split times is provided it must have proper size.
        # If it is not provided the info will be lost and only the new 
        # splits will be saved at the end of the propagation.
        if not splitTimesList:
            splitTimesList = [[]]*len(set)
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
                   splitTimes = splitTimesList[i])
            
            out.append(temp)
        # element at the top of the stack is stored in a variable
        self._stack=temp
    
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
            if (self._checkStep):
                # if the integration step was accepted check for split
                temp_f = ADS(
                         self._stack.ADSPatch.box, 
                         self._stack.ADSPatch.nsplit, 
                         self._runningX)
                
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
                xf)

        # if not at end of propagation this means that a split was 
        # necessary (and possible) from how we defined the checkevent
        if (not self._reachstime):

            print(
                "The domain was split at instant ",
                self._input.t,
                " and continued the propagation...")
            
            # compute split subdomains
            dir=state.direction()
            Dl, Dr = state.split(dir)
            
            # update list of split times for each subdomain
            tempSplitTimes = self._stack.splitTimes.copy()
            tempSplitTimes.append(self._input.t)

            # add elements to the list of objects that need propagation
            tempL = ADSstate(
                    Dl, 
                    self._input.t, 
                    self._input.h, 
                    splitTimes = tempSplitTimes)
            tempR = ADSstate(
                    Dr, 
                    self._input.t, 
                    self._input.h, 
                    splitTimes = tempSplitTimes)
            
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
                    splitTimes = self._stack.splitTimes)
            
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
        return None
    
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
        
        return

    def propagate(
        self, 
        set: List[ADS], 
        t1: float, 
        t2: float,
        splitTimesList: Optional(List[List[float]]) = [],
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
        """

        """ 
        NB: t1 and t2 are not used in this function. They are only left
        here to maintain same syntax as the parent class method that 
        this function overloads. The initial time of the propagation 
        MUST be set through integrator.loadTime and it will be adapted
        online by each patch. Conversely, tf is fixed and it MUST be
        set through integrator.loadTime.
        """
        
        # list of final domains
        listOut : List[ADSstate] = [] 
        # create initial list from initial set of domains
        listIn = self._InitializeList(set, splitTimesList)

        print("Let's start propagation...")
        # loop until list of domains is not empty
        while listIn:
            # remove first domain and put it at the top of the stack
            self._importfromList(listIn)
            # loop until final time is reached for the domain
            while (not self._reachstime):
                xf = super(ADSintegrator, self).propagate(self._stack.ADSPatch.manifold, self._input.t, t2)
                # after each call to propagation update list of patches
                listIn, listOut = self._SplitStateProcess(xf, listIn, listOut)

        print("Propagation completed. ")

        return listOut


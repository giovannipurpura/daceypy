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

from typing import (List, Tuple, Union, Callable, Optional,
                    overload)

import numpy as np
from numpy.typing import NDArray

from daceypy import DA, array

from ._PrettyType import PrettyType


class ADS(metaclass=PrettyType):
    """
    ADS base element: a map (vector or scalar) and its splitting history
    """
    def __init__(self, box: array, nsplit: NDArray(np.int_), manifold: Optional[array] = None):
        self.box = np.atleast_1d(box)
        self.nsplit = np.atleast_1d(nsplit)
        if manifold is None:
            self.manifold = np.atleast_1d(box)
        else:
            self.manifold = np.atleast_1d(manifold)
        
    
    # function to split the domain
    def split_domain(self, direction: int) -> Tuple[ADS, ADS]:
        """
        Splits the current domain into two new domains and returns them

        Args:
            direction : index of the DA variable along which domain has to be split
        
        Returns:
            Two ned domains as instances of ADS class.

        Raises:
            DACEException
        """
        x_esp = self.box
        y_esp = self.manifold
        splits1 = self.nsplit
        splits2 = self.nsplit

        d1 = array.identity()
        d2 = array.identity()

        d1[direction] = 0.5 * DA(direction + 1) - 0.5
        
        splits1=np.append(splits1,-(direction + 1))

        d2[direction] = 0.5 * DA(direction + 1) + 0.5
        splits2=np.append(splits2,(direction + 1))

        x1 = x_esp.eval(d1)
        x2 = x_esp.eval(d2)

        y1 = y_esp.eval(d1)
        y2 = y_esp.eval(d2)

        return ADS(x1,splits1,y1), ADS(x2,splits2,y2)

    #can the domain be split or has the maximum number of splits been reached?
    def cansplit(self, N_max: int) -> bool:
        """
        Splits the current domain into two new domains and returns them

        Args:
            direction : index of the DA variable along which domain has to be split
        
        Returns:
            True if the domain can be split, False otherwise.

        Raises:
            DACEException
        """
        return np.sum(self.count_splits()) < N_max

    #Count number of previous splits by direction
    def count_splits(self)-> NDArray(int):
        """
        counts the number of split that the current domain has undergone for each dimension.

        Args:
        
        Returns:
            split_dir: True if the domain can be split, False otherwise.

        Raises:
            DACEException
        """
        #Take from element the n of splits
        splits = self.nsplit
        #n of splits in each direction
        nvar=DA.getMaxVariables()
        split_dir=np.zeros((nvar,))

        for i in range(splits.size):
            num = int(abs(splits[i]))
            split_dir[num - 1] = split_dir[num - 1] + 1
        return split_dir

    # decide the split direction:
    def __direction_mono(self, map_1d: DA, type_: int = 0) -> int:
        """
        Computes the splitting direction for a single DA map, returns the index of the variable to split.

        Args:
            map_1d: DA variable containing the map to be split.
            type_: type of the norm to be used to estimate truncation order

        Returns:
            dir: The index of the variable along which the split occurs.

        Raises:
            DACEException
        
        See also:
            DA.estimNorm

        """
        nvar = DA.getMaxVariables()
        ord = map_1d.getTO()
        err = 0.
        dir = 0
        for i in range(nvar):
            estim, _ = map_1d.estimNorm(i + 1, type_, ord + 1)
            err_new = estim[estim.size - 1]
            if (err_new > err):
                err = err_new
                dir = i
        return dir

    def direction(self, type_: int = 0) -> int:
        """
        Return a integer with the split direction: this is the direction for the element of the input vector producing the worst error

        Args:
            type_: type of the norm to be used.
        
        Returns:
            The index of the variable along which the split occurs.

        Raises:
            DACEException

        See also:
            DA.estimNorm
        """
        errors = self.manifold.getTruncationErrors(type_)
        maxLocation = np.argmax(errors)
        return self.__direction_mono(self.manifold[maxLocation], type_)


    #Check if the domain need to be split based on tolerance
    def __check_split_mono(self, map_1d: DA, type_: int = 0, toll: float = 0.) -> bool:
        """
        Checks whether this DA expansion need to be split

        Args:
            type_: type of norm to be computed, see estimNorm.
            toll : tolerance to determine split necessity

        Returns:
            A bool to indicate whether it is necessary to split or not.

        Raises:
            DACEException
        See also:
            DA.estimNorm

        """
        ord = map_1d.getTO()
        estim, _ = map_1d.estimNorm(0, type_, ord + 1)
        err = estim[estim.size - 1]
        if (err > toll):
            return True
        return False

    @overload
    def check_split(self, toll: float = 0., type_: int = 0) -> bool:
        """
        Return a check whether this vector map needs to be split according to a maximum threshold along all possible elements of the input DA vector

        Args:
            type_: type of the norm to be used, see documentation for DA.estimNorm.
            toll : tolerance determining the split

        Returns:
            A bool to indicate whether it is necessary to split or not.

        Raises:
            DACEException
        
        See also:
            DA.estimNorm
        """
        ...

    @overload
    def check_split(self, toll: NDArray[np.double], type_: int = 0) -> int:
        """
        Return a check whether this vector map need to be split according to a set of thresholds, one for each elements of the input DA vector

        Args:
            type_: type of the norm to be used, see documentation for DA.estimNorm.
            toll : tolerance determining the split
        
        Returns:
            A bool to indicate whether it is necessary to split or not.

        Raises:
            DACEException
        
        See also:
            DA.estimNorm
        """
        ...

    def check_split(self, toll: Union[float, NDArray[np.double]], type_: int = 0) -> bool:
        """
        Return a check whether this vector map need to be split

        Args:
            type_: type of the norm to be used, see documentation for DA.estimNorm.
            toll : tolerance determining the split (can be either a scalar or a vector)

        Raises:
            DACEException
            ValueError: if toll has size different from the input vector
            TypeError: if toll is not a float or array of floats
        
        See also:
            DA.estimNorm
        """
        b_errors = self.manifold.getTruncationErrors(type_)
        if isinstance(toll, float): # ecc....
            errors=b_errors
            TOL = toll*np.ones(errors.shape)
        elif isinstance(toll, array):
            if b_errors.size != toll.size:
                raise ValueError('Tolerance must have the same size as the number of dimension of the input vector')
            errors = b_errors-toll
            errors [errors < 0.] = 0.
            TOL = toll
        else:
            raise TypeError('toll can only be a float or an array of floats')
        
        maxError = np.amax(errors)
        maxLocation = np.argmax(errors)

        return self.__check_split_mono(self.manifold[maxLocation],type_, TOL[maxLocation])

    # Base routine for ADS
    @staticmethod
    def ADSroutine(initial_domains: List[ADS], toll: Union[float, NDArray[np.double]], N_max: int, eFun: Callable[[ADS], ADS], type_: int = 0)-> List[ADS]:
        """
        Base implementation of ADS. Starts from the list of initial domains and evaluates the transformation function for each one.
        When the domain need to be split, the new subdomains are added to the initial list of domains that need to be transformed.
        When the final list of domains do not need to be further split (or cannot be further split) the process ends

        Args:
            initial_domains: initial list of domains expressed as instances of ADS class
            toll : tolerance determining the split (can be either a scalar or a vector)
            N_max: maximum number of times that the domain can be split, does not limit the max number of domains which in principle is 2^N_max
            eFun: transformation function which each domain need to undergo, must take an input an instance of ADS class and must return an instance of ADS class
            type_: type of the norm to be used during split, see documentation for DA.estimNorm.
        
        See also:
            DA.estimNorm
        """
        domains = initial_domains.copy()
        final_domains = []
        while domains: # loop untill all subdomains have not been propagated and checked
            d_el = domains.pop() # remove first element from the list of domains that need to be addressed and pass it through the objective function
            print("-------------Element evaluation---------------")
            p_el = eFun(d_el)
            print("-------------------Done-----------------------" )

            if p_el.check_split(toll, type_):
                if p_el.cansplit(N_max): #  if the maximum number of splits has not been reached split the domain and append the new ones to the ones that need to be analyzed
                    dir = p_el.direction(type_) # checks if domain needs to be split, if it does it returns a direction
                    Dl, Dr = d_el.split_domain(dir)
                    domains.extend([Dl,Dr])
                    print("A split occurred, total number of domains that need propagation are now: ", domains.__len__())
                else: # if the maximum number of splits for this domain has been reached just mark it as final and signal possible inaccuracies
                    print ("A domain needed to split but reached maximum number of splits: possible inaccuracies may arise!")
                    final_domains.append(p_el)
            else: # if the domain does not need to be further split:
                final_domains.append(p_el)
                print("one domain reached end of transformation")
            
            print("remaining domains to be transformed: ", domains.__len__())
        
        return final_domains
















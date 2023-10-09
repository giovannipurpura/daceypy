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

from typing import Callable, List, Optional, Tuple, Union, overload

import numpy as np
from numpy.typing import NDArray

from daceypy import DA, array

from ._PrettyType import PrettyType


class ADS(metaclass=PrettyType):
    """
    ADS (Adaptive Domain Splitting) base element: a map (vector or scalar) and its splitting history.
    """
    def __init__(self, box: array, nsplit: List[int] = [], manifold: Optional[array] = None):
        """
        Initialize an ADS object.

        Args:
            box:
            nsplit: splitting history, i.e. list of integers for which splittings happened
            manifold:
        """
        self.box = array(np.atleast_1d(box))
        self.nsplit = nsplit
        self.manifold = array(np.atleast_1d(box if manifold is None else manifold))

    def split(self, direction: int) -> Tuple[ADS, ADS]:
        """
        Split the current domain into two new domains and return them.

        Args:
            direction: index of the DA variable along which domain has to be split

        Returns:
            Two new domains as instances of ADS class.

        Raises:
            DACEException
        """
        x_esp = self.box
        y_esp = self.manifold
        splits1 = self.nsplit.copy()
        splits2 = self.nsplit.copy()

        d1 = array.identity()
        d2 = array.identity()

        d1[direction] = 0.5 * DA(direction + 1) - 0.5
        splits1.append(-(direction + 1))

        d2[direction] = 0.5 * DA(direction + 1) + 0.5
        splits2.append(+(direction + 1))

        x1 = x_esp.eval(d1)
        x2 = x_esp.eval(d2)

        y1 = y_esp.eval(d1)
        y2 = y_esp.eval(d2)

        return ADS(x1, splits1, y1), ADS(x2, splits2, y2)

    def canSplit(self, N_max: int) -> bool:
        """
        Check if the domain can be split.

        Args:
            N_max: maximum number of allowed splits.

        Returns:
            True if the domain can be split, False otherwise.

        Raises:
            DACEException
        """
        return np.sum(self.countSplits()) < N_max

    def countSplits(self) -> NDArray[np.int_]:
        """
        Count the number of splits that the current domain has undergone for each dimension.

        Returns:
            Number of splits for each dimension as NDArray.

        Raises:
            DACEException
        """
        # Take from element the n of splits
        splits = self.nsplit
        # n of splits in each direction
        nvar = DA.getMaxVariables()
        split_dir = np.zeros((nvar, ), np.int_)

        for split in splits:
            num = abs(split)
            split_dir[num - 1] += 1

        return split_dir

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
        max_location = np.argmax(errors)

        map_1d: DA = self.manifold[max_location]
        nvar = DA.getMaxVariables()
        ord = map_1d.getTO()
        err = 0.0
        dir = 0
        for i in range(nvar):
            estim, _ = map_1d.estimNorm(i + 1, type_, ord + 1)
            err_new = estim[estim.size - 1]
            if err_new > err:
                err = err_new
                dir = i
        return dir

    @overload
    def checkSplit(self, toll: float = 0.0, type_: int = 0) -> bool:
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
    def checkSplit(self, toll: NDArray[np.double], type_: int = 0) -> int:
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

    def checkSplit(self, toll: Union[float, NDArray[np.double]], type_: int = 0) -> bool:
        """
        Check whether this vector map needs to be split.

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

        if isinstance(toll, float):
            errors = b_errors
            toll_vec = np.full(errors.shape, toll)
        elif isinstance(toll, np.ndarray):
            if b_errors.size != toll.size:
                raise ValueError('toll must have the same size as the number of dimension of the input vector')
            errors = b_errors - toll
            errors[errors < 0.0] = 0.0
            toll_vec = toll
        else:
            raise TypeError('toll can only be a float or an array of floats')

        max_location = np.argmax(errors)

        map_1d = self.manifold[max_location]
        toll = toll_vec[max_location]

        ord = map_1d.getTO()
        estim, _ = map_1d.estimNorm(0, type_, ord + 1)
        err = estim[estim.size - 1]

        return err > toll

    @staticmethod
    def eval(
        initial_domains: List[ADS],
        toll: Union[float, NDArray[np.double]],
        N_max: int,
        fun: Callable[[ADS], ADS],
        type_: int = 0,
        log_fun: Callable[[str]] = print,
    ) -> List[ADS]:
        """
        Apply a transformation function to a list of ADS domains.

        When a domain needs to be split, the new subdomains are added to the initial list of domains that needs to be transformed.
        When the final list of domains does not need further splitting (or cannot be split any more) the process ends,

        Args:
            initial_domains:
                initial list of domains expressed as instances of ADS class
            toll:
                tolerance determining the split (can be either a scalar or a vector)
            N_max:
                maximum number of times that the domain can be split,
                does not limit the max number of domains which in principle is 2^N_max
            fun:
                transformation function which each domain needs to undergo,
                must take as input an instance of ADS class and must return an instance of ADS class
            type_:
                type of the norm to be used during split, see documentation for DA.estimNorm
            log_fun:
                function invoked for progress and status logging,
                to disable logging pass a no-op function (e.g., `lambda s: None`)

        See also:
            DA.estimNorm
        """

        log_fun("Starting ADS evaluation with", len(initial_domains), "domains")
        domains = initial_domains.copy()
        final_domains: List[ADS] = []
        # loop until all subdomains have not been propagated and checked
        while domains:
            # remove first element from the list of domains that need to be addressed
            d_el = domains.pop()
            log_fun("Evaluating transformation function on a domain...")
            # evaluate transformation function on the domain
            p_el = fun(d_el)
            log_fun("Done." )

            # check if the domain can still be split
            if p_el.checkSplit(toll, type_):
                # checks if domain needs to be split
                if p_el.canSplit(N_max):
                    # compute splitting direction
                    dir = p_el.direction(type_)
                    # split the domain along the direction
                    new_domains = d_el.split(dir)
                    # append the new domains to the list
                    domains.extend(new_domains)
                    log_fun("A split occurred, total number of domains that need propagation are now:", len(domains))
                else:
                    # maximum number of splits for this domain has been reached,
                    # just mark it as final and signal possible inaccuracies
                    log_fun("A domain needed to split, but the maximum number of splits has been reached: possible inaccuracies may arise!")
                    final_domains.append(p_el)
            else:
                # no further splitting needed
                final_domains.append(p_el)
                log_fun("One domain reached end of transformation. Still to process:", len(domains), "domains.")

            log_fun("Remaining domains to be transformed:", len(domains))

        log_fun("Final number of domains:", len(final_domains))

        return final_domains

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

from ctypes import c_double, c_uint

from . import _DA
from ._PrettyType import PrettyType


class Monomial(metaclass=PrettyType):

    def __init__(self):
        self.m_jj = (c_uint * _DA.DA.getMaxVariables())()
        self.m_coeff = c_double()

    def order(self) -> int:
        """
        Compute the order of the monomial.

        Returns:
            Order of the monomial

        Derived from C++:
            `unsigned int Monomial::order()`
        """
        return sum(self.m_jj)

    def toString(self) -> str:
        """
        Convert monomial to string.

        Returns:
            A string representing the monomial in human readable form.

        Derived from C++:
            `std::string Monomial::toString()`
        """
        oss = []
        oss.append("     I  COEFFICIENT              ORDER EXPONENTS\n")
        oss.append("     1  ")
        oss.append(f"{self.m_coeff.value:24.16e}")
        oss.append(f"{self.order():4d} ")
        for e in self.m_jj:
            oss.append(f" {e:2d}")
        oss.append("\n------------------------------------------------")
        return "".join(oss)

    __str__ = toString

    def __repr__(self):
        return f"<Monomial of order {self.order()}>"

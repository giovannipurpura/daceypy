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

from typing import Dict, Optional, overload

from . import core
from ._PrettyType import PrettyType


class DACEException(Exception, metaclass=PrettyType):

    @overload
    def __init__(self) -> None:
        """
        Create a DACEException object from an existing severity and error codes.

        Derived from C++:
            `DACEException::DACEException()`
        """
        ...

    @overload
    def __init__(self, exc_sv: int, exc_id: int) -> None:
        """
        Create a DACEException object from given severity and ID codes.

        Args:
            exc_sv: severity code of the error.
            exc_id: ID code of the error.

        Derived from C++:
            `DACEException::DACEException(const int exc_sv, const int exc_id)`
        """
        ...

    def __init__(
        self,
        exc_sv: Optional[int] = None,
        exc_id: Optional[int] = None,
    ) -> None:

        super().__init__()

        if exc_sv is not None:
            if exc_id is None:
                raise TypeError("exc_sv and exc_id must be defined together")
            if not isinstance(exc_sv, int):
                raise TypeError("exc_sv must be an int")
            if not isinstance(exc_id, int):
                raise TypeError("exc_id must be an int")
            self.m_x = exc_sv
            self.m_yy = exc_id
        elif exc_id is not None:
            raise TypeError("exc_sv and exc_id must be defined together")
        else:
            self.m_x = core.GetErrorX()
            self.m_yy = core.GetErrorYY()

        self.updateMessage()

    def __str__(self):
        return self.msg

    # *************************************************************************
    # *     Private member functions
    # *************************************************************************

    def updateMessage(self):
        """
        Update the error message of this exception based on its ID.
        """

        self.id = self.m_x * 100 + self.m_yy

        if self.m_x > 10:
            self.msg = f"{DACEerr[self.id]} (ID: {self.id})"
        else:
            self.msg = f"{core.GetErrorMSG().decode()} (ID: {self.id})"
            core.ClearError()


DACEerr: Dict[int, str] = {
    0: "DACE: Unknown DACE error. Contact Dinamica SRL for filing a bug report.",

    # 101: "DACEDAL: Attempt to deallocate protected or unallocated variable, ignored",
    # 102: "DACESETNOT: Truncation order set to value larger then maximum order, setting to maximum order",
    # 103: "DACEEST: Not enough non-zero monomials found, returned estimate may be inaccurate",
    # 104: "DACEREAD: Line numbering out of order, ignoring line numbers",
    # 105: "DACEREAD: DA vector contains more variables than current setup, ignoring coefficient",
    # 106: "DACEREAD: Duplicate monomial in DA vector",
    # 107: "DACEINI: Requested order has been increased to the minimum required order of 1",
    # 108: "DACEINI: Requested number of variables has been increased to the minimum required number of variables 1",

    # 601: "DACEVAR: Requested independent variable is out of bounds of current setup, returning zero DA",
    # 602: "DACEPOK: Not enough storage to insert monomials, truncating",
    # 603: "DACEVAR: Not enough storage to set up variable, returning zero DA",
    # 604: "DACECOEF: Not enough storage to set coefficient, truncating",
    # 605: "DACEDIVC: Divide by zero, returning zero DA",
    # 606: "DACEINT: Requested independent variable out of bounds of current setup, returning zero DA",
    # 607: "DACEDER: Requested independent variable out of bounds of current setup, returning zero DA",
    # 608: "DACEMINV: Divide by zero, returning zero DA",
    # 609: "DACESQRT: Negative constant part in square root, returning zero DA",
    # 610: "DACEISQRT: Negative constant part in inverse square root, returning zero DA",
    # 611: "DACELOG: Negative constant part in logarithm, returning zero DA",
    # 612: "DACETAN: Cosine is zero in tangent, returning zero DA",
    # 613: "DACEASIN: Constant part is out of domain [-1,1] in arcsine, returning zero DA",
    # 614: "DACEACOS: Constant part is out of domain [-1,1] in arccosine, returning zero DA",
    # 617: "DACEACOSH: Constant part is out of domain [1,infinity) in hyperbolic arccosine, returning zero DA",
    # 618: "DASEATANH: Constant part is out of domain [-1,1], returning zero DA",
    # 619: "DACEONORM: Requested independent variable out of bounds of current setup, returning zero",
    # 620: "DACEEST: Maximum order must be at least 2 in order to use DA estimation",
    # 622: "DACEPLUG: Requested independent variable out of bounds of current setup, returning zero DA",
    # 623: "DACELOGB: Logarithm base must be positive, returning zero DA",
    # 624: "DACEREAD: Not enough lines provided to read a DA vector, returning zero DA",
    # 625: "DACEREAD: Unrecognized DA input format, returning zero DA",
    # 627: "DACEENC: Invalid exponents with order larger than maximum order, returning zero",
    # 628: "DACEDEC: Invalid DA codes provided, returning zero",
    # 629: "DACEROOT: Zero-th root does not exists, returning zero DA",
    # 630: "DACEROOT: Negative or zero constant part in even root, returning zero DA",
    # 631: "DACEROOT: Zero constant part in odd root, returning zero DA",
    # 632: "DACEPAC: Not enough storage in the target object, truncating",
    # 633: "DACECMUL: Not enough storage in the target object, truncating",
    # 634: "DACELIN: Not enough storage in the target object, truncating",
    # 635: "DACEINT: Not enough storage in the target object, truncating",
    # 636: "DACEDER: Not enough storage in the target object, truncating",
    # 637: "DACECOP: Not enough storage in the target object, truncating",
    # 638: "DACEPUSH: Not enough space to store the provided data, truncating",
    # 639: "DACEPULL: Not enough space to store the requested data, truncating",
    # 640: "DACETRIM: Not enough space to store the requested data, truncating",
    # 641: "DACENORM: Unknown norm type, resorting to max norm",
    # 642: "DACEONORM: Unknown norm type, resorting to max norm",
    # 643: "DACETRIM: Not enough storage in the target object, truncating",
    # 644: "DACEWRITE: DA vector contains more monomials than expected, truncating",

    # 901: "DACEINI: Internal array size exceeds available addressing space",

    # 98: "DACEINI: Internal error, size of generated DA coding arrays does not match number of monomials",
    # 97: "DACEINI: Internal error, generated DA coding arrays are faulty",
    # 96: "DACEINI: Internal error, unable to correctly allocate scratch variables",
    # 94: "DACEINI: Internal error, memory allocation failure",
    # 93: "DACEREALL: Internal error, memory allocation failure",
    # 92: "DACEALL: DACE has not been initialized",
    # 90: "DACEINFO: DA Object not allocated",

    101: "DA::getCoeff: Not enough exponents, missing exponents treated as zero",
    102: "DA::setCoeff: Not enough exponents, missing exponents treated as zero",
    103: "DA::getCoeff: More exponents than variables, ignoring extra exponents",
    104: "DA::setCoeff: More exponents than variables, ignoring extra exponents",

    604: "compiledDA::compiledDA(): Dimension lower than 1",
    605: "compiledDA::eval: Argument size lower than the number of variables in the polynomial",

    99: "DA::checkVersion: DACE C++ interface header file and DACE core library version mismatch",
}

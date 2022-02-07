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


class PrettyType(type):
    """
    This class is used to have a better representation
    when type() is called on a DACEyPy object or the __module__ is accessed.
    """

    def __new__(cls, name, bases, classdict):
        t = super().__new__(cls, name, bases, classdict)
        t.__module__ = "daceypy"
        return t

    def __repr__(self):
        return "daceypy." + self.__name__

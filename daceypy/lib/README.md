# DACEyPy dynamic-link libraries

This folder contains the dynamic-link libraries used by DACEyPy.
Libraries are loaded by [../core.py](../core.py).

## Recompile the DACE core

The core is compiled from DACE (https://github.com/dacelib/dace),
by removing non-core libraries (dacecxx and dacecxx_s)
(it is not necessary, it is just for reducing the file size).

To do so, in the file `interfaces/cxx/CMakeLists.txt`
every reference to those libraries is removed.

## Reference version

The version of DACE used to compile the core is:
https://github.com/dacelib/dace/tree/2bda904d411d915b94bdf99d84209875a3243178

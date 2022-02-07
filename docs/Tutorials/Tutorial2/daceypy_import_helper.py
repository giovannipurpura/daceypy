try:
    # check if daceypy is installed in the system
    import daceypy
except ImportError:
    # load daceypy from the folder
    import sys
    from inspect import getsourcefile
    from pathlib import Path
    sourcefile = getsourcefile(lambda: 0)
    if sourcefile is None:
        exit("Cannot load daceypy")
    sys.path.append(str(Path(sourcefile).resolve().parents[3]))
    import daceypy

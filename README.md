## PyPower package

This is a python package oriented towards the simulation of large interconnected power grids, with a chance of line failures occuring when a given line energy goes beyond some threshold.

Once pulled, the package can be run as long as mpi4py and numpy are installed. Simply run:
```
mpirun ./pypower [inifile]
```

where the `inifile` is a path to the ini file giving the parameters which run the simulation. Some examples are included here, for example the first thing to try is `mpirun ./pypower tests/test1.ini`



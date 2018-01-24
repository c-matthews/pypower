## PyPower package

This is a python package oriented towards the simulation of large interconnected power grids, with a chance of line failures occuring when a given line energy goes beyond some threshold.

Once pulled, the package can be run as long as mpi4py and numpy are installed. Simply run:
```
mpirun ./pypower [inifile]
```

where the `inifile` is a path to the ini file giving the parameters which run the simulation. Some examples are included here, for example the first thing to try is `mpirun ./pypower tests/test1.ini`. The .ini files include all of the parameters needed for a simulation, including the model data, as well as scenario info and output locations. The example tests are included, all of them using the larger (149 node) model. The examples should run quickly when in parallel with `mpirun`. The tests are listed below:

* tests/test1.ini :: Runs a single 100 second simulation of the power network, with the failure set so high that no lines will fail in such a short time window. The trajectory information is put into the tests/output1/ directory
* tests/test2.ini :: Runs one hundred simulations of the power network, where a line fails if it hits 8.5% of its susceptance value. Each simulation continues until the load served drops below 50%.
<<<<<<< Updated upstream
* tests/test3.ini :: Runs one hundred simulations of the network with line 164 taken down after the initial burn-in period. The simulations end once the second line has gone down. The raw trajectory information is not saved, instead a list of events is saved.
* tests/test4.ini :: Runs one hundred Parallel Replica simulations with line 164 taken down initially. Simulations continue until 0 load is served (i.e. total failure). A list of events is saved in the output directory.
=======
* tests/test3.ini :: Runs one hundred simulations of the network with line 164 taken down after the initial burn-in period. The simulations end once the second line has gone down. The raw trajectory information is not saved, instead a list of events is saved. 
>>>>>>> Stashed changes

We recommend adapting these ini file examples as needed.


<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes

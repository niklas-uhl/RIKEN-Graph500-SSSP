# License
* Since the codes in ./src/generator/ are from the reference implemenation (https://graph500.org/), the license of them is the same.
* The license of the other codes is Apache License v2.0.

# SSSP

Contains:

SSSP solver for Graph500 benchmark: sssp

Sequential SSSP solver (intended mostly for testing): sssp-seq

## Build

Assuming you have cmake installed, run the following commands.

```
mkdir build
cd build
cmake ..
make
```

On Fugaku use:
```
cmake -DMY_SYSTEM="Fugaku" ..
```


NOTE: Currently the below parameters have to be hard-set in apps/CMakeLists.txt (todo make user parameter!)

```

* `VERBOSE` : toggle verbose output. true = enable, false = disenable.
* `VERTEX_REORDERING` : specify vertex reordering mode. 0 = do nothing (default), 1 = only reduce isolated vertices, 2 = sort by degree and reduce isolated vertices.
* `REAL_BENCHMARK` : change SSSP iteration times. true = 64 times, false = 16 times (for testing).
```

## Test

NOTE: does not work yet, TODO

```
ctest --verbose
```

## Run

Best to use n^2 processes, for some natural n.

To set delta value x (between 0 and 1) for delta-stepping, set:

```sh
export DELTA_STEP=x
```


Simple run:

```sh
export OMP_NUM_THREADS=<nthreads>

mpirun -np <nprocs> ./bin/sssp-parallel <nscale>
```

With output files:

```sh
# OpenMPI >= 1.8
mpirun -np <nprocs> -bind-to none -output-filename ./log/lP1T8S16VR0BNONE -x OMP_NUM_THREADS=<nthreads> ./bin/sssp-parallel <nscale>
# OpenMPI <= 1.6.5
mpirun -np <nprocs> -bind-to-none -output-filename ./log/lP1T8S16VR0BNONE -x OMP_NUM_THREADS=<nthreads> ./bin/sssp-parallel <nscale>
# MPICH / MVAPICH
```

* `nprocs` : number of processes
* `nthreads` : number of threads
* `nscale` :   exponent for number of vertices (2^nscale)

```

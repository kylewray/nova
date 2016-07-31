The 'nova' Library
====

CUDA optimized code for solving MDPs, POMDPs, and Dec-POMDPs.

A tutorial will be added soon.

If you use this library, then please cite our AAAI 2015 Fall Symposium paper:

Wray, Kyle H. and Zilberstein, Shlomo. “A Parallel Point-Based POMDP Algorithm Leveraging GPUs.” AAAI Fall Symposium on Sequential Decision Making for Intelligent Agents (SDMIA), Arlington, Virginia, USA, November 2015.

## Build

First, install gcc and the CUDA toolkit which contains nvcc. To build the library, navigate to the root and run:
```
make clean -i ; make
```

## Test

The nova library contains a few kinds of tests. Unit tests for C/C++/CUDA, also computing code coverage, ensure the core code works properly. Implementation tests consist of grid world and traditional research baseline domains.

Note that executing unit or implementation tests may require ```optirun``` or other similar programs, since it uses the GPU.

### Unit Tests

These test only the C/C++/CUDA code (currently). Building and executing the test suite can be done all at once via:
```
make tests
```

Optionally, it can be done in stages. First, build the test library (which has code coverage flags for gcc):
```
make novat
```
Next, build the unit test suites:
```
make nova_tests
```
Finally, run the test suites:
```
make run_tests
```

### Implementation Tests

These test both the C/C++/CUDA code as well as the Python code. Examples include:
```
python3 tests/grid_world/grid_world.py
python3 tests/tiger/tiger.py
```

## Benchmarks

Benchmarks to compare algorithm performance can be found in the ```tests/benchmarks``` directory. Examples include:
```
python3 tests/benchmarks/algorithms/algorithms.py
python3 tests/benchmarks/parallel/parallel.py
```
The performance of each algorithm, in terms of run time, initial state value, and average reward (e.g., ADR) in trials is reported in the resultant ```results``` folders inside the benchmark's directory.

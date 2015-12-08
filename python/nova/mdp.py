""" The MIT License (MIT)

    Copyright (c) 2015 Kyle Hollins Wray, University of Massachusetts

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
    the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
    FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
    COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import time

import ctypes as ct
import numpy as np

import csv

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))
import nova_mdp as nm
import file_loader as fl


class MDP(nm.NovaMDP):
    """ A Markov Decision Process (MDP) object that can load, solve, and save.

        Specifically, it is capable of loading raw and cassandra-like MDP files, provides
        functionality to solve them using the nova library, and enables saving the resulting
        policy as a raw policy file.
    """

    def __init__(self):
        """ The constructor for the MDP class. """

        # Assign a nullptr for the device-side pointers. These will be set if the GPU is utilized.
        self.ng = int(0)
        self.goals = ct.POINTER(ct.c_uint)()
        self.currentHorizon = int(0)
        self.V = ct.POINTER(ct.c_float)()
        self.VPrime = ct.POINTER(ct.c_float)()
        self.pi = ct.POINTER(ct.c_uint)()
        self.ne = int(0)
        self.expanded = ct.POINTER(ct.c_int)()
        self.d_goals = ct.POINTER(ct.c_uint)()
        self.d_S = ct.POINTER(ct.c_int)()
        self.d_T = ct.POINTER(ct.c_float)()
        self.d_R = ct.POINTER(ct.c_float)()
        self.d_V = ct.POINTER(ct.c_float)()
        self.d_VPrime = ct.POINTER(ct.c_float)()
        self.d_pi = ct.POINTER(ct.c_uint)()
        self.d_expanded = ct.POINTER(ct.c_int)()

        # Additional informative variables.
        self.Rmin = None
        self.Rmax = None
        self.epsilon = 0.01

    def load(self, filename, filetype='cassandra', scalarize=lambda x: x[0]):
        """ Load a Multi-Objective POMDP file given the filename and optionally the file type.

            Parameters:
                filename    --  The name and path of the file to load.
                filetype    --  Either 'cassandra' or 'raw'. Default is 'cassandra'.
                scalarize   --  Optionally define a scalarization function. Only used for 'raw' files.
                                Default returns the first reward.
        """

        fileLoader = fl.FileLoader()

        if filetype == 'cassandra':
            fileLoader.load_cassandra(filename)
        elif filetype == 'raw':
            fileLoader.load_raw_mdp(filename, scalarize)
        else:
            print("Invalid file type '%s'." % (filetype))
            raise Exception()

        self.n = fileLoader.n
        self.ns = fileLoader.ns
        self.m = fileLoader.m

        self.s0 = fileLoader.s0
        self.ng = fileLoader.ng

        self.gamma = fileLoader.gamma
        self.horizon = fileLoader.horizon
        self.epsilon = fileLoader.epsilon

        self.Rmin = fileLoader.Rmin
        self.Rmax = fileLoader.Rmax

        array_type_ng_uint = ct.c_uint * (self.ng)
        array_type_nmns_int = ct.c_int * (self.n * self.m * self.ns)
        array_type_nmns_float = ct.c_float * (self.n * self.m * self.ns)
        array_type_nm_float = ct.c_float * (self.n * self.m)

        self.goals = array_type_ng_uint(*fileLoader.goals.flatten())
        self.S = array_type_nmns_int(*fileLoader.S.flatten())
        self.T = array_type_nmns_float(*fileLoader.T.flatten())
        self.R = array_type_nm_float(*fileLoader.R.flatten())

    def solve(self, algorithm='vi', process='gpu', numThreads=1024, heuristic=None):
        """ Solve the MDP using the nova Python wrapper.

            Parameters:
                algorithm   --  The algorithm to use, either 'vi', 'lao*', or 'rtdp'. Default is 'vi'.
                process     --  Use the 'cpu' or 'gpu'. If 'gpu' fails, it tries 'cpu'. Default is 'gpu'.
                numThreads  --  The number of CUDA threads to execute (multiple of 32). Default is 1024.
                heuristic   --  For 'lao*', this function or list maps state indexes to heuristic values.
                                Optional. Default value is None, yielding an n-array of zeros.

            Returns:
                V, pi, timing           -- If algorithm is 'vi'.
                r, S, V, pi, timing     -- If algorithm is 'lao*'.
                r       --  The size of the valid states found by heuristic search algorithms (e.g., 'lao*').
                S       --  The actual r state indexes found by heuristic search algorithms (e.g., 'lao*').
                V       --  The values of each state, mapping states to values. In heuristic search, S contains the state index.
                pi      --  The policy, mapping states to actions. In heuristic search, S contains the state index.
                timing  --  A pair (wall-time, cpu-time) for solver execution time, not including (un)initialization.
        """

        # Create V and pi, assigning them their respective initial values.
        Vinitial = np.array([0.0 for s in range(self.n)])
        if self.gamma < 1.0:
            Vinitial = np.array([float(self.Rmin / (1.0 - self.gamma)) for s in range(self.n)])

        # Create functions to convert flattened NumPy arrays to C arrays.
        array_type_n_float = ct.c_float * self.n

        # Create a C array for the initial values.
        Vinitial = array_type_n_float(*Vinitial)

        # Create C arrays for the result. Some are unused depending on the algorithm.
        r = ct.c_uint(0)
        S = ct.POINTER(ct.c_uint)()
        V = ct.POINTER(ct.c_float)()
        pi = ct.POINTER(ct.c_uint)()

        # For informed search algorithms, define the heuristic, which is stored in V initially.
        if algorithm == 'lao*' and heuristic is not None:
            self.V = array_type_n_float(*np.array([float(heuristic[s]) for s in range(self.n)]))

        timing = None

        # If the process is 'gpu', then attempt to solve it. If an error arises, then
        # assign process to 'cpu' and attempt to solve it using that.
        if process == 'gpu':
            timing = (time.time(), time.clock())
            if algorithm == 'vi':
                result = nm._nova.mdp_vi_complete_gpu(self, int(numThreads), Vinitial,
                                                            ct.byref(V), ct.byref(pi))
            elif algorithm == 'lao*':
                result = nm._nova.ssp_lao_star_complete_gpu(self, int(numThreads), Vinitial,
                                                            ct.byref(r), ct.byref(S),
                                                            ct.byref(V), ct.byref(pi))
            elif algorithm == 'rtdp':
                result = nm._nova.ssp_rtdp_complete_gpu(self, int(numThreads), Vinitial,
                                                            ct.byref(r), ct.byref(S),
                                                            ct.byref(V), ct.byref(pi))
            timing = (time.time() - timing[0], time.clock() - timing[1])

            if result != 0:
                print("Failed to execute the 'nova' library's GPU MDP solver.")
                process = 'cpu'

        # If the process is 'cpu', then attempt to solve it.
        if process == 'cpu':
            timing = (time.time(), time.clock())
            if algorithm == 'vi':
                result = nm._nova.mdp_vi_complete_cpu(self, Vinitial, V, pi)
            elif algorithm == 'lao*':
                result = nm._nova.ssp_lao_star_complete_cpu(self, Vinitial, r, S, V, pi)
            elif algorithm == 'rtdp':
                result = nm._nova.ssp_rtdp_complete_cpu(self, Vinitial, r, S, V, pi)
            timing = (time.time() - timing[0], time.clock() - timing[1])

            if result != 0:
                print("Failed to execute the 'nova' library's CPU MDP solver.")
                raise Exception()

        if result == 0:
            if algorithm == 'vi':
                V = np.array([V[i] for i in range(self.n)])
                pi = np.array([pi[i] for i in range(self.n)])
            elif algorithm == 'lao*' or algorithm == 'rtdp':
                r = r.value
                S = np.array([S[i] for i in range(r)])
                V = np.array([V[i] for i in range(r)])
                pi = np.array([pi[i] for i in range(r)])
        else:
            print("Failed to solve MDP using the 'nova' library.")
            raise Exception()

        if algorithm == 'vi':
            return V, pi, timing
        elif algorithm == 'lao*' or algorithm == 'rtdp':
            return r, S, V, pi, timing

    def __str__(self):
        """ Return the string of the MDP values akin to the raw file format.

            Returns:
                The string of the MDP in a similar format as the raw file format.
        """

        result = "n:       " + str(self.n) + "\n"
        result += "m:       " + str(self.m) + "\n"
        result += "ns:      " + str(self.ns) + "\n"
        result += "s0:      " + str(self.s0) + "\n"
        result += "goals:   " + str([self.goals[i] for i in range(self.ng)]) + "\n"
        result += "horizon: " + str(self.horizon) + "\n"
        result += "gamma:   " + str(self.gamma) + "\n\n"

        result += "S(s, a, s'):\n%s" % (str(np.array([self.S[i] \
                    for i in range(self.n * self.m * self.ns)]).reshape((self.n, self.m, self.ns)))) + "\n\n"

        result += "T(s, a, s'):\n%s" % (str(np.array([self.T[i] \
                    for i in range(self.n * self.m * self.ns)]).reshape((self.n, self.m, self.ns)))) + "\n\n"

        result += "R(s, a):\n%s" % (str(np.array([self.R[i] \
                    for i in range(self.n * self.m)]).reshape((self.n, self.m)))) + "\n\n"

        return result


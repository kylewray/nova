""" The MIT License (MIT)

    Copyright (c) 2016 Kyle Hollins Wray, University of Massachusetts

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

import file_loader as fl
import mdp_value_function as mvf

import nova_mdp as nm


class MDP(nm.NovaMDP):
    """ A Markov Decision Process (MDP) object that can load, solve, and save.

        Specifically, it is capable of loading raw and cassandra-like MDP files, provides
        functionality to solve them using the nova library, and enables saving the resulting
        policy as a raw policy file.
    """

    def __init__(self):
        """ The constructor for the MDP class. """

        # Assign a nullptr for the device-side pointers and initial values for the structure variables.
        self.n = int(0)
        self.ns = int(0)
        self.m = int(0)
        self.gamma = float(0.9)
        self.horizon = int(1)
        self.epsilon = float(0.01)
        self.s0 = int(0)
        self.ng = int(0)
        self.goals = ct.POINTER(ct.c_uint)()
        self.S = ct.POINTER(ct.c_int)()
        self.T = ct.POINTER(ct.c_float)()
        self.R = ct.POINTER(ct.c_float)()
        self.d_goals = ct.POINTER(ct.c_uint)()
        self.d_S = ct.POINTER(ct.c_int)()
        self.d_T = ct.POINTER(ct.c_float)()
        self.d_R = ct.POINTER(ct.c_float)()

        # Additional useful variables not in the structure.
        self.Rmin = None
        self.Rmax = None

        self.cpuIsInitialized = False
        self.gpuIsInitialized = False

    def __del__(self):
        """ The deconstructor for the MDP class. """

        self.uninitialize_gpu()
        self.uninitialize()

    def initialize(self, n, ns, m, gamma, horizon, epsilon, s0, ng):
        """ Initialize the MDP object, allocating array memory, given the parameters.

        Parameters:
            n       --  The number of states.
            ns      --  The maximum number of successor states.
            m       --  The number of actions.
            gamma   --  The discount factor between 0 and 1.
            horizon --  The positive integer for the horizon.
            epsilon --  The convergence criterion for some algorithms.
            s0      --  The initial state index (if an SSP MDP).
            ng      --  The positive integer for number of goals (if an SSP MDP) or 0 (otherwise).
        """

        if self.cpuIsInitialized:
            return

        result = nm._nova.mdp_initialize_cpu(self, n, ns, m, gamma, horizon, epsilon, s0, ng)
        if result != 0:
            print("Failed to initialize the MDP.")
            raise Exception()

        self.cpuIsInitialized = True

    def uninitialize(self):
        """ Uninitialize the MDP object, freeing the allocated memory. """

        if not self.cpuIsInitialized:
            return

        result = nm._nova.mdp_uninitialize_cpu(self)
        if result != 0:
            print("Failed to uninitialize the MDP.")
            raise Exception()

        self.cpuIsInitialized = False

    def initialize_gpu(self):
        """ Initialize the GPU variables. This only needs to be called if GPU algorithms are used. """

        if self.gpuIsInitialized:
            return

        result = nm._nova.mdp_initialize_successors_gpu(self)
        result += nm._nova.mdp_initialize_state_transitions_gpu(self)
        result += nm._nova.mdp_initialize_rewards_gpu(self)

        if self.ng > 0:
            result += nm._nova.mdp_initialize_goals_gpu(self)

        if result != 0:
            print("Failed to initialize the 'nova' library's GPU variables for the MDP.")
            raise Exception()

        self.gpuIsInitialized = True

    def uninitialize_gpu(self):
        """ Uninitialize the GPU variables. This only needs to be called if GPU algorithms are used. """

        if not self.gpuIsInitialized:
            return

        result = nm._nova.mdp_uninitialize_successors_gpu(self)
        result += nm._nova.mdp_uninitialize_state_transitions_gpu(self)
        result += nm._nova.mdp_uninitialize_rewards_gpu(self)

        if self.ng > 0:
            result += nm._nova.mdp_uninitialize_goals_gpu(self)

        if result != 0:
            print("Failed to uninitialize the 'nova' library's GPU variables for the MDP.")
            raise Exception()

        self.gpuIsInitialized = False

    def load(self, filename, filetype='cassandra', scalarize=lambda x: x[0]):
        """ Load a Multi-Objective POMDP file given the filename and optionally the file type.

            Parameters:
                filename    --  The name and path of the file to load.
                filetype    --  Either 'cassandra' or 'raw'. Default is 'cassandra'.
                scalarize   --  Optionally define a scalarization function. Only used for 'raw' files.
                                Default returns the first reward.
        """

        # Before anything, uninitialize the current MDP.
        self.uninitialize_gpu()
        self.uninitialize()

        # Now load the file based on the desired file type.
        fileLoader = fl.FileLoader()

        if filetype == 'cassandra':
            fileLoader.load_cassandra(filename)
        elif filetype == 'raw':
            fileLoader.load_raw_mdp(filename, scalarize)
        else:
            print("Invalid file type '%s'." % (filetype))
            raise Exception()

        # Allocate the memory on the C-side. Note: Allocating on the Python-side will create managed pointers.
        self.initialize(fileLoader.n, fileLoader.ns, fileLoader.m,
                        fileLoader.gamma, fileLoader.horizon, fileLoader.epsilon,
                        fileLoader.s0, fileLoader.ng)

        # Flatten all of the file loader data.
        fileLoader.goals = fileLoader.goals.flatten()
        fileLoader.S = fileLoader.S.flatten()
        fileLoader.T = fileLoader.T.flatten()
        fileLoader.R = fileLoader.R.flatten()

        # Copy all of the variables' data into these arrays.
        for i in range(self.ng):
            self.goals[i] = fileLoader.goals[i]
        for i in range(self.n * self.m * self.ns):
            self.S[i] = fileLoader.S[i]
            self.T[i] = fileLoader.T[i]
        for i in range(self.n * self.m):
            self.R[i] = fileLoader.R[i]

        self.Rmin = fileLoader.Rmin
        self.Rmax = fileLoader.Rmax

    # TODO: REMOVE THIS. IT HAS BEEN REPLACED BY SEPARATE CLASSES.
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

        # Create a function to convert a flattened numpy arrays to a C array, then convert Vinitial.
        array_type_n_float = ct.c_float * self.n
        Vinitial = array_type_n_float(*Vinitial)

        # For informed search algorithms, define the heuristic, which is stored in V initially.
        if algorithm == 'lao*' and heuristic is not None:
            self.V = array_type_n_float(*np.array([float(heuristic[s]) for s in range(self.n)]))

        policy = ct.POINTER(mvf.MDPValueFunction)()
        timing = None

        # If the process is 'gpu', then attempt to solve it. If an error arises, then
        # assign process to 'cpu' and attempt to solve it using that.
        if process == 'gpu':
            timing = (time.time(), time.clock())
            if algorithm == 'vi':
                result = nm._nova.mdp_vi_complete_gpu(self, int(numThreads), Vinitial,
                                                            ct.byref(policy))
            elif algorithm == 'lao*':
                result = nm._nova.ssp_lao_star_complete_gpu(self, int(numThreads), Vinitial,
                                                            ct.byref(policy))
            elif algorithm == 'rtdp':
                result = nm._nova.ssp_rtdp_complete_gpu(self, int(numThreads), Vinitial,
                                                            ct.byref(policy))
            timing = (time.time() - timing[0], time.clock() - timing[1])

            if result != 0:
                print("Failed to execute the 'nova' library's GPU MDP solver.")
                process = 'cpu'

        # If the process is 'cpu', then attempt to solve it.
        if process == 'cpu':
            timing = (time.time(), time.clock())
            if algorithm == 'vi':
                result = nm._nova.mdp_vi_complete_cpu(self, Vinitial, ct.byref(policy))
            elif algorithm == 'lao*':
                result = nm._nova.ssp_lao_star_complete_cpu(self, Vinitial, ct.byref(policy))
            elif algorithm == 'rtdp':
                result = nm._nova.ssp_rtdp_complete_cpu(self, Vinitial, ct.byref(policy))
            timing = (time.time() - timing[0], time.clock() - timing[1])

            if result != 0:
                print("Failed to execute the 'nova' library's CPU MDP solver.")
                raise Exception()

        # Dereference the pointer (this is how you do it in ctypes).
        policy = policy.contents

        return policy, timing

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


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

import ctypes as ct
import platform
import os.path

import csv
import numpy as np


# Import the correct library file depending on the platform.
_nova = None
if platform.system() == "Windows":
    _nova = ct.CDLL(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                    "..", "..", "lib", "nova.dll"))
else:
    _nova = ct.CDLL(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                    "..", "..", "lib", "nova.so"))


_nova.mdp_vi_cpu_complete.argtypes = (ct.c_uint,                # n
                                        ct.c_uint,              # m
                                        ct.c_uint,              # ns
                                        ct.POINTER(ct.c_int),   # S
                                        ct.POINTER(ct.c_float), # T
                                        ct.POINTER(ct.c_float), # R
                                        ct.c_float,             # gamma
                                        ct.c_uint,              # horizon
                                        ct.POINTER(ct.c_float), # V
                                        ct.POINTER(ct.c_uint))  # pi

_nova.mdp_vi_gpu_complete.argtypes = (ct.c_uint,                # n
                                        ct.c_uint,              # m
                                        ct.c_uint,              # ns
                                        ct.POINTER(ct.c_int),   # S
                                        ct.POINTER(ct.c_float), # T
                                        ct.POINTER(ct.c_float), # R
                                        ct.c_float,             # gamma
                                        ct.c_uint,              # horizon
                                        ct.c_uint,              # numThreads
                                        ct.POINTER(ct.c_float), # V
                                        ct.POINTER(ct.c_uint))  # pi


def mdp_vi_cpu_complete(n, m, ns, S, T, R, gamma, horizon, V, pi):
    """ The wrapper Python function for executing value iteration for an MDP using the CPU.

        Parameters:
            n           --  The number of states.
            m           --  The number of actions.
            ns          --  The maximum number of successor states.
            S           --  The successor states as a flattened 3-dimensional array (n-m-ns-array).
            T           --  The state transitions as a flattened 3-dimensional array (n-m-ns-array).
            R           --  The reward function as a flattened 2-dimensional array (n-m-array).
            gamma       --  The discount factor.
            horizon     --  The number of iterations to execute.
            V           --  The resultant values of the states (n-array). Modified.
            pi          --  The resultant actions to take at each state (n-array). Modified.

        Returns:
            Zero on success; non-zero otherwise.
    """

    global _nova

    array_type_nmns_int = ct.c_int * (int(n) * int(m) * int(ns))
    array_type_nmns_float = ct.c_float * (int(n) * int(m) * int(ns))
    array_type_nm_float = ct.c_float * (int(n) * int(m))
    array_type_n_float = ct.c_float * int(n)
    array_type_n_uint = ct.c_uint * int(n)

    VResult = array_type_n_float(*V)
    piResult = array_type_n_uint(*pi)

    result = _nova.mdp_vi_cpu_complete(int(n), int(m), int(ns),
                            array_type_nmns_int(*S), array_type_nmns_float(*T), array_type_nm_float(*R),
                            float(gamma), int(horizon),
                            VResult, piResult)

    if result == 0:
        for i in range(n):
            V[i] = VResult[i]
            pi[i] = piResult[i]

    return result


def mdp_vi_gpu_complete(n, m, ns, S, T, R, gamma, horizon, numThreads, V, pi):
    """ The wrapper Python function for executing value iteration for an MDP using the GPU.

        Parameters:
            n           --  The number of states.
            m           --  The number of actions.
            ns          --  The maximum number of successor states.
            S           --  The successor states as a flattened 3-dimensional array (n-m-ns-array).
            T           --  The state transitions as a flattened 3-dimensional array (n-m-ns-array).
            R           --  The reward function as a flattened 2-dimensional array (n-m-array).
            gamma       --  The discount factor.
            horizon     --  The number of iterations to execute.
            numThreads  --  The number of CUDA threads to execute.
            V           --  The resultant values of the states (n-array). Modified.
            pi          --  The resultant actions to take at each state (n-array). Modified.

        Returns:
            Zero on success; non-zero otherwise.
    """

    global _nova

    array_type_nmns_int = ct.c_int * (int(n) * int(m) * int(ns))
    array_type_nmns_float = ct.c_float * (int(n) * int(m) * int(ns))
    array_type_nm_float = ct.c_float * (int(n) * int(m))
    array_type_n_float = ct.c_float * int(n)
    array_type_n_uint = ct.c_uint * int(n)

    VResult = array_type_n_float(*V)
    piResult = array_type_n_uint(*pi)

    result = _nova.mdp_vi_gpu_complete(int(n), int(m), int(ns),
                            array_type_nmns_int(*S), array_type_nmns_float(*T), array_type_nm_float(*R),
                            float(gamma), int(horizon), int(numThreads),
                            VResult, piResult)

    if result == 0:
        for i in range(n):
            V[i] = VResult[i]
            pi[i] = piResult[i]

    return result


class MOMDP(object):
    """ A Multi-Objective Markov Decision Process (MOMDP) object that can load, solve, and save.

        Specifically, it is capable of loading raw and cassandra-like MDP files, provides
        functionality to solve them using the nova library, and enables saving the resulting
        policy as a raw policy file.
    """

    def __init__(self):
        """ The constructor for the MOMDP class. """

        self.n = 0          # The number of states.
        self.m = 0          # The number of actions.
        self.ns = 0         # The maximum number of successor states.
        self.k = 1          # The number of reward functions.

        self.S = list()     # Successor states.
        self.T = list()     # State transitions T(s, a, s').
        self.R = list()     # Rewards R_i(s, a).

        self.s0 = 0         # The initial state.
        self.horizon = 0    # The horizon; non-positive numbers imply infinite horizon.
        self.gamma = 0.9    # The discount factor.

    def load(self, filename):
        """ Load a raw MOMDP file given the file.

            Parameters:
                filename    --  The name and path of the file to load.
        """

        # Load all the data in this object.
        data = list()
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                data += [list(row)]

        # Attempt to parse all the data into their respective variables.
        try:
            self.n = int(data[0][0])
            self.m = int(data[0][1])
            self.ns = int(data[0][2])
            self.k = int(data[0][3])

            self.s0 = int(data[0][4])
            self.horizon = int(data[0][5])
            self.gamma = float(data[0][6])

            rowOffset = 1
            self.S = np.array([[[int(data[(self.n * a + s) + rowOffset][sp]) \
                                for sp in range(self.ns)] \
                            for a in range(self.m)] \
                        for s in range(self.n)])

            rowOffset = 1 + self.n * self.m
            self.T = np.array([[[float(data[(self.n * a + s) + rowOffset][sp]) \
                                for sp in range(self.ns)] \
                            for a in range(self.m)] \
                        for s in range(self.n)])

            rowOffset = 1 + self.n * self.m + self.n * self.m
            self.R = np.array([[[float(data[(self.m * i + a) + rowOffset][s])
                                for a in range(self.m)] \
                            for s in range(self.n)] \
                        for i in range(self.k)])

        except Exception:
            print("Failed to load file '%s'." % (filename))
            raise Exception()

    def solve(self, f=None, numThreads=1024):
        """ Solve the MOMDP, using the nova python wrapper, with the given scalarization function.

            Parameters:
                f           --  The scalarization function which maps a reward vector to a single
                                reward. If set to None, then only the first reward function is
                                used. Default is None. Optional.
                numThreads  --  The number of CUDA threads to execute (multiple of 32). Default is 1024.

            Returns:
                V   --  The values of each state, mapping states to values.
                pi  --  The policy, mapping states to actions.
        """

        if f == None:
            # The initial value of V is either Rmin / (1 - gamma), for gamma less than 1.0, and
            # simply 0.0 otherwise.
            V = None
            if self.gamma < 1.0:
                Rmin = np.array(self.R[0]).min()
                V = np.array([float(Rmin / (1.0 - self.gamma)) for s in range(self.n)])
            else:
                V = np.array([0.0 for s in range(self.n)])

            pi = np.array([0 for s in range(self.n)])

            numThreads = 1024

            # Call the nova library to run value iteration. If the GPU is available, then use it.
            # Otherwise, use the CPU.
            result = mdp_vi_gpu_complete(self.n, self.m, self.ns,
                            self.S.flatten(), self.T.flatten(), self.R[0].flatten(),
                            self.gamma, self.horizon, numThreads,
                            V, pi)
            #result = mdp_vi_cpu_complete(self.n, self.m, self.ns,
            #                self.S.flatten(), self.T.flatten(), self.R[0].flatten(),
            #                self.gamma, self.horizon,
            #                V, pi)
            if result != 0:
                print("Failed to execute nova's solver.")

            return V, pi

        return None

    def __str__(self):
        """ Return the string of the MOMDP values akin to the raw file format.

            Returns:
                The string of the MOMDP in a similar format as the raw file format.
        """

        result = "n:       " + str(self.n) + "\n"
        result += "m:       " + str(self.m) + "\n"
        result += "ns:      " + str(self.ns) + "\n"
        result += "k:       " + str(self.k) + "\n"
        result += "s0:      " + str(self.s0) + "\n"
        result += "horizon: " + str(self.horizon) + "\n"
        result += "gamma:   " + str(self.gamma) + "\n\n"

        result += "S(s, a, i):\n%s" % (str(np.array(self.S))) + "\n\n"
        result += "T(s, a, s'):\n%s" % (str(np.array(self.T))) + "\n\n"

        for i in range(self.k):
            result += "R_%i(s, a):\n%s" % (i + 1, str(np.array(self.R[i]))) + "\n\n"

        return result


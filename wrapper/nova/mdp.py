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


class NovaMDP(ct.Structure):
    """ The C struct MDP object. """

    _fields_ = [("n", ct.c_uint),
                ("ns", ct.c_uint),
                ("m", ct.c_uint),
                ("gamma", ct.c_float),
                ("horizon", ct.c_uint),
                ("S", ct.POINTER(ct.c_int)),
                ("T", ct.POINTER(ct.c_float)),
                ("R", ct.POINTER(ct.c_float)),
                ("currentHorizon", ct.c_uint),
                ("V", ct.POINTER(ct.c_float)),
                ("VPrime", ct.POINTER(ct.c_float)),
                ("pi", ct.POINTER(ct.c_uint)),
                ("d_S", ct.POINTER(ct.c_int)),
                ("d_T", ct.POINTER(ct.c_float)),
                ("d_R", ct.POINTER(ct.c_float)),
                ("d_V", ct.POINTER(ct.c_float)),
                ("d_VPrime", ct.POINTER(ct.c_float)),
                ("d_pi", ct.POINTER(ct.c_uint)),
                ]


_nova.mdp_vi_complete_cpu.argtypes = (ct.POINTER(NovaMDP),
                                    ct.POINTER(ct.c_float), # V
                                    ct.POINTER(ct.c_uint))  # pi

_nova.mdp_vi_complete_gpu.argtypes = (ct.POINTER(NovaMDP),
                                    ct.c_uint,              # numThreads
                                    ct.POINTER(ct.c_float), # V
                                    ct.POINTER(ct.c_uint))  # pi


def mdp_vi_complete_cpu(n, ns, m, gamma, horizon, S, T, R, V, pi):
    """ The wrapper Python function for executing value iteration for an MDP using the CPU.

        Parameters:
            n           --  The number of states.
            ns          --  The maximum number of successor states.
            m           --  The number of actions.
            gamma       --  The discount factor.
            horizon     --  The number of iterations to execute.
            S           --  The successor states as a flattened 3-dimensional array (n-m-ns-array).
            T           --  The state transitions as a flattened 3-dimensional array (n-m-ns-array).
            R           --  The reward function as a flattened 2-dimensional array (n-m-array).
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

    # Note: The 'ct.POINTER(ct.c_xyz)()' below simply assigns a nullptr value in the struct
    # to the corresponding value. This device pointer will never be assigned for the CPU version.
    result = _nova.mdp_vi_complete_cpu(NovaMDP(int(n), int(ns), int(m),
                                float(gamma), int(horizon),
                                array_type_nmns_int(*S),
                                array_type_nmns_float(*T),
                                array_type_nm_float(*R),
                                int(0),
                                ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_uint)(),
                                ct.POINTER(ct.c_int)(),
                                ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_uint)()),
                            VResult, piResult)

    if result == 0:
        for i in range(n):
            V[i] = VResult[i]
            pi[i] = piResult[i]

    return result


def mdp_vi_complete_gpu(n, ns, m, S, T, R, gamma, horizon, numThreads, V, pi):
    """ The wrapper Python function for executing value iteration for an MDP using the GPU.

        Parameters:
            n           --  The number of states.
            ns          --  The maximum number of successor states.
            m           --  The number of actions.
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

    # Note: The 'ct.POINTER(ct.c_xyz)()' below simply assigns a nullptr value in the struct
    # to the corresponding value. This device pointer will be assigned later.
    result = _nova.mdp_vi_complete_gpu(NovaMDP(int(n), int(ns), int(m),
                                float(gamma), int(horizon),
                                array_type_nmns_int(*S),
                                array_type_nmns_float(*T),
                                array_type_nm_float(*R),
                                int(0),
                                ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_uint)(),
                                ct.POINTER(ct.c_int)(),
                                ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_uint)()),
                            int(numThreads), VResult, piResult)

    if result == 0:
        for i in range(n):
            V[i] = VResult[i]
            pi[i] = piResult[i]

    return result


class MDP(NovaMDP):
    """ A Markov Decision Process (MDP) object that can load, solve, and save.

        Specifically, it is capable of loading raw and cassandra-like MDP files, provides
        functionality to solve them using the nova library, and enables saving the resulting
        policy as a raw policy file.
    """

    def __init__(self):
        """ The constructor for the MDP class. """

        # Assign a nullptr for the device-side pointers. These will be set if the GPU is utilized.
        self.currentHorizon = int(0)
        self.V = ct.POINTER(ct.c_float)()
        self.VPrime = ct.POINTER(ct.c_float)()
        self.pi = ct.POINTER(ct.c_uint)()
        self.d_S = ct.POINTER(ct.c_int)()
        self.d_T = ct.POINTER(ct.c_float)()
        self.d_R = ct.POINTER(ct.c_float)()
        self.d_V = ct.POINTER(ct.c_float)()
        self.d_VPrime = ct.POINTER(ct.c_float)()
        self.d_pi = ct.POINTER(ct.c_uint)()

        # Additional informative variables.
        self.Rmin = None

        # Optional variables to be used for solving the SSP version.
        self.s0 = None
        self.goals = None

    def load(self, filename, scalarize=lambda x: x[0]):
        """ Load a raw Multi-Objective MDP file given the filename and optionally a scalarization function.

            Parameters:
                filename    --  The name and path of the file to load.
                scalarize   --  Optionally define a scalarization function. Default returns the first reward.
        """

        # Load all the data in this object.
        data = list()
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                data += [list(row)]

        # Attempt to parse all the data into their respective variables.
        try:
            # Load the header information.
            self.n = int(data[0][0])
            self.ns = int(data[0][1])
            self.m = int(data[0][2])

            k = int(data[0][3])
            self.s0 = int(data[0][4])

            self.horizon = int(data[0][5])
            self.gamma = float(data[0][6])

            # Functions to convert flattened NumPy arrays to C arrays.
            array_type_nmns_int = ct.c_int * (self.n * self.m * self.ns)
            array_type_nmns_float = ct.c_float * (self.n * self.m * self.ns)
            array_type_nm_float = ct.c_float * (self.n * self.m)

            # Load each of the larger data structures into memory and immediately
            # convert them to their C object type to save memory.
            rowOffset = 1
            self.S = array_type_nmns_int(*np.array([[[int(data[(self.n * a + s) + rowOffset][sp]) \
                                for sp in range(self.ns)] \
                            for a in range(self.m)] \
                        for s in range(self.n)]).flatten())

            rowOffset = 1 + self.n * self.m
            self.T = array_type_nmns_float(*np.array([[[float(data[(self.n * a + s) + rowOffset][sp]) \
                                for sp in range(self.ns)] \
                            for a in range(self.m)] \
                        for s in range(self.n)]).flatten())

            rowOffset = 1 + self.n * self.m + self.n * self.m
            self.R = array_type_nm_float(*scalarize(np.array([[[float(data[(self.m * i + a) + rowOffset][s])
                                for a in range(self.m)] \
                            for s in range(self.n)] \
                        for i in range(k)])).flatten())

            self.Rmin = min([self.R[i] for i in range(self.n * self.m)])

        except Exception:
            print("Failed to load file '%s'." % (filename))
            raise Exception()

    def solve(self, numThreads=1024):
        """ Solve the MDP using the nova Python wrapper.

            Parameters:
                numThreads  --  The number of CUDA threads to execute (multiple of 32). Default is 1024.

            Returns:
                V   --  The values of each state, mapping states to values.
                pi  --  The policy, mapping states to actions.
        """

        global _nova

        # Create V and pi, assigning them their respective initial values.
        V = np.array([0.0 for s in range(self.n)])
        if self.gamma < 1.0:
            V = np.array([float(self.Rmin / (1.0 - self.gamma)) for s in range(self.n)])
        pi = np.array([0 for s in range(self.n)])

        # Create functions to convert flattened NumPy arrays to C arrays.
        array_type_n_float = ct.c_float * self.n
        array_type_n_uint = ct.c_uint * self.n

        # Create C arrays for the result.
        VResult = array_type_n_float(*V)
        piResult = array_type_n_uint(*pi)

        # Solve the MDP using the nova library and return the solution. If it
        # fails to use the GPU version, then it tries the CPU version.
        result = _nova.mdp_vi_complete_gpu(self, int(numThreads), VResult, piResult)
        if result != 0:
            result = _nova.mdp_vi_complete_cpu(self, VResult, piResult)

        if result == 0:
            V = np.array([VResult[i] for i in range(self.n)])
            pi = np.array([piResult[i] for i in range(self.n)])
        else:
            print("Failed to solve MDP using the 'nova' library.")
            raise Exception()

        return V, pi

    def __str__(self):
        """ Return the string of the MDP values akin to the raw file format.

            Returns:
                The string of the MDP in a similar format as the raw file format.
        """

        result = "n:       " + str(self.n) + "\n"
        result += "m:       " + str(self.m) + "\n"
        result += "ns:      " + str(self.ns) + "\n"
        result += "s0:      " + str(self.s0) + "\n"
        result += "horizon: " + str(self.horizon) + "\n"
        result += "gamma:   " + str(self.gamma) + "\n\n"

        result += "S(s, a, s'):\n%s" % (str(np.array([self.S[i] \
                    for i in range(self.n * self.m * self.ns)]).reshape((self.n, self.m, self.ns)))) + "\n\n"

        result += "T(s, a, s'):\n%s" % (str(np.array([self.T[i] \
                    for i in range(self.n * self.m * self.ns)]).reshape((self.n, self.m, self.ns)))) + "\n\n"

        result += "R(s, a):\n%s" % (str(np.array([self.R[i] \
                    for i in range(self.n * self.m)]).reshape((self.n, self.m)))) + "\n\n"

        return result


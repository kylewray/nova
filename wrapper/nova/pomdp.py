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
import itertools


# Import the correct library file depending on the platform.
_nova = None
if platform.system() == "Windows":
    _nova = ct.CDLL(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                    "..", "..", "lib", "nova.dll"))
else:
    _nova = ct.CDLL(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                    "..", "..", "lib", "nova.so"))


_nova.pomdp_pbvi_complete_gpu.argtypes = (ct.c_uint,            # n
                                        ct.c_uint,              # ns
                                        ct.c_uint,              # m
                                        ct.c_uint,              # z
                                        ct.c_uint,              # r
                                        ct.c_uint,              # rz
                                        ct.POINTER(ct.c_int),   # Z
                                        ct.POINTER(ct.c_float), # B
                                        ct.POINTER(ct.c_int),   # S
                                        ct.POINTER(ct.c_float), # T
                                        ct.POINTER(ct.c_float), # O
                                        ct.POINTER(ct.c_float), # R
                                        ct.c_float,             # gamma
                                        ct.c_uint,              # horizon
                                        ct.c_uint,              # numThreads
                                        ct.POINTER(ct.c_float), # Gamma
                                        ct.POINTER(ct.c_uint))  # pi


def pomdp_pbvi_complete_gpu(n, ns, m, z, r, rz,
        Z, B, S, T, O, R,
        gamma, horizon, numThreads, Gamma, pi):
    """ The wrapper Python function for executing point-based value iteration for a POMDP.

        Parameters:
            n                   --  The number of states.
            ns                  --  The maximum number of successor states.
            m                   --  The number of actions.
            z                   --  The number of observations.
            r                   --  The number of belief points.
            rz                  --  The maximum number of non-zero belief values over all beliefs.
            Z                   --  The belief-state pairs as a flattened 2-dimensional array.
            B                   --  The belief points as a flattened 2-dimensional array.
            S                   --  The state-action pairs as a flattened 2-dimensional array.
            T                   --  The state transitions as a flattened 3-dimensional array.
            O                   --  The observation transitions as a flattened 3-dimensional array.
            R                   --  The reward function as a flattened 2-dimensional array.
            gamma               --  The discount factor.
            horizon             --  The number of iterations.
            numThreads          --  The number of CUDA threads to execute.
            Gamma               --  The resultant alpha-vectors. Modified.
            pi                  --  The resultant actions for each alpha-vector. Modified.

        Returns:
            Zero on success, and a non-zero nova error code otherwise.
    """

    global _nova

    array_type_rrz_int = ct.c_int * (int(r) * int(rz))
    array_type_rrz_float = ct.c_float * (int(r) * int(rz))
    array_type_nmns_int = ct.c_int * (int(n) * int(m) * int(ns))
    array_type_nmns_float = ct.c_float * (int(n) * int(m) * int(ns))
    array_type_mnz_float = ct.c_float * (int(m) * int(n) * int(z))
    array_type_nm_float = ct.c_float * (int(n) * int(m))
    array_type_rn_float = ct.c_float * (int(r) * int(n))
    array_type_r_uint = ct.c_uint * int(r)

    GammaResult = array_type_rn_float(*Gamma)
    piResult = array_type_r_uint(*pi)

    result = _nova.pomdp_pbvi_complete_gpu(int(n), int(ns), int(m), int(z), int(r), int(rz),
                            array_type_rrz_int(*Z), array_type_rrz_float(*B),
                            array_type_nmns_int(*S), array_type_nmns_float(*T),
                            array_type_mnz_float(*O), array_type_nm_float(*R),
                            float(gamma), int(horizon), int(numThreads),
                            GammaResult, piResult)

    if result == 0:
        for i in range(r * n):
            Gamma[i] = GammaResult[i]
        for i in range(r):
            pi[i] = piResult[i]

    return result


class MOPOMDP(object):
    """ A Multi-Objective Partially Observable Markov Decision Process (MOPOMDP) object
        that can load, solve, and save.

        Specifically, it is capable of loading raw and cassandra-like POMDP files, provides
        functionality to solve them using the nova library, and enables saving the resulting
        policy as a raw policy file.
    """

    def __init__(self):
        """ The constructor for the MOPOMDP class. """

        self.n = 0  # The number of states.
        self.ns = 0 # The maximum number of successors over all states.
        self.m = 0  # The number of actions.
        self.z = 0  # The number of observations.
        self.r = 0  # The number of belief points.
        self.rz = 0 # The maximum number of non-zero values in a belief point.
        self.k = 1  # The number of reward functions.

        self.S = list() # A mapping from states to actions to the list of successors.
        self.T = list() # State transitions T(s, a, s').
        self.O = list() # Observation transitions T(a, s', o).
        self.R = list() # Rewards R_i(s, a).

        self.Z = list() # The mapping of beliefs to non-zero index values.
        self.B = list() # The initial set of belief states.

        self.horizon = 0 # The horizon; non-positive numbers imply infinite horizon.
        self.gamma = 0.9 # The discount factor.

    def load(self, filename):
        """ Load a raw MOPOMDP file given the file.

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
            self.ns = int(data[0][1])
            self.m = int(data[0][2])
            self.z = int(data[0][3])
            self.r = int(data[0][4])
            self.rz = int(data[0][5])
            self.k = int(data[0][6])

            self.horizon = int(data[0][7])
            self.gamma = float(data[0][8])

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
            self.O = np.array([[[float(data[(self.z * a + o) + rowOffset][sp]) \
                                for o in range(self.z)] \
                            for sp in range(self.n)] \
                        for a in range(self.m)])

            rowOffset = 1 + self.n * self.m + self.n * self.m + self.m * self.z
            self.R = np.array([[[float(data[(self.m * i + a) + rowOffset][s])
                                for a in range(self.m)] \
                            for s in range(self.n)] \
                        for i in range(self.k)])

            rowOffset = 1 + self.n * self.m + self.n * self.m + self.m * self.z + self.k * self.m
            self.Z = np.array([[int(data[i + rowOffset][s])
                            for s in range(self.rz)] \
                        for i in range(self.r)])

            rowOffset = 1 + self.n * self.m + self.n * self.m + self.m * self.z + self.k * self.m + self.r
            self.B = np.array([[float(data[i + rowOffset][s])
                            for s in range(self.rz)] \
                        for i in range(self.r)])

        except Exception:
            print("Failed to load file '%s'." % (filename))
            raise Exception()

    def solve(self, f=None, numThreads=1024):
        """ Solve the MOPOMDP, using the nova python wrapper, with the given scalarization function.

            Parameters:
                f           --  The scalarization function which maps a reward vector to a single
                                reward. If set to None, then only the first reward function is
                                used. Default is None. Optional.
                numThreads  --  The number of CUDA threads to execute.

            Returns:
                Gamma   --  The alpha-vectors, one for each belief point, mapping states to values.
                pi      --  The policy, mapping alpha-vectors (belief points) to actions.
        """

        if f == None:
            # The initial alpha-vectors are either Rmin / (1 - gamma), for gamma less than 1.0, and
            # simply 0.0 otherwise.
            Gamma = None
            if self.gamma < 1.0:
                Rmin = np.array(self.R[0]).min()
                Gamma = np.array([[float(Rmin / (1.0 - self.gamma)) for s in range(self.n)] \
                            for i in range(self.r)])
            else:
                Gamma = np.array([[0.0 for s in range(self.n)] for b in range(self.r)])
            Gamma = Gamma.flatten()

            pi = np.array([0 for i in range(self.r)])

            numThreads = 1024

            # Call the nova library to run value iteration.
            result = pomdp_pbvi_complete_gpu(self.n, self.ns, self.m, self.z, self.r, self.rz,
                        self.Z.flatten(), self.B.flatten(), self.S.flatten(), self.T.flatten(), self.O.flatten(), self.R[0].flatten(),
                        self.gamma, self.horizon, numThreads,
                        Gamma, pi)
            if result != 0:
                print("Failed to execute nova's solver.")

            Gamma = Gamma.reshape((self.r, self.n))

            return Gamma, pi

        return None

    def __str__(self):
        """ Return the string of the MOPOMDP values akin to the raw file format.

            Returns:
                The string of the MOPOMDP in a similar format as the raw file format.
        """

        result = "n:       " + str(self.n) + "\n"
        result += "ns:      " + str(self.ns) + "\n"
        result += "m:       " + str(self.m) + "\n"
        result += "z:       " + str(self.z) + "\n"
        result += "r:       " + str(self.r) + "\n"
        result += "rz:      " + str(self.rz) + "\n"
        result += "k:       " + str(self.k) + "\n"
        result += "horizon: " + str(self.horizon) + "\n"
        result += "gamma:   " + str(self.gamma) + "\n\n"

        result += "S:\n%s" % (str(np.array(self.S))) + "\n\n"
        result += "T(s, a, s'):\n%s" % (str(np.array(self.T))) + "\n\n"
        result += "O(a, s', o):\n%s" % (str(np.array(self.O))) + "\n\n"

        for i in range(self.k):
            result += "R_%i(s, a):\n%s" % (i + 1, str(np.array(self.R[i]))) + "\n\n"

        result += "Z:\n%s" % (str(np.array(self.Z))) + "\n\n"
        result += "B:\n%s" % (str(np.array(self.B))) + "\n\n"

        return result


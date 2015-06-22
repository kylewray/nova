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


class NovaPOMDP(ct.Structure):
    """ The C struct POMDP object. """

    _fields_ = [("n", ct.c_uint),
                ("ns", ct.c_uint),
                ("m", ct.c_uint),
                ("z", ct.c_uint),
                ("r", ct.c_uint),
                ("rz", ct.c_uint),
                ("gamma", ct.c_float),
                ("horizon", ct.c_uint),
                ("S", ct.POINTER(ct.c_int)),
                ("T", ct.POINTER(ct.c_float)),
                ("O", ct.POINTER(ct.c_float)),
                ("R", ct.POINTER(ct.c_float)),
                ("Z", ct.POINTER(ct.c_int)),
                ("B", ct.POINTER(ct.c_float)),
                ("currentHorizon", ct.c_uint),
                ("Gamma", ct.POINTER(ct.c_float)),
                ("GammaPrime", ct.POINTER(ct.c_float)),
                ("pi", ct.POINTER(ct.c_uint)),
                ("d_S", ct.POINTER(ct.c_int)),
                ("d_T", ct.POINTER(ct.c_float)),
                ("d_O", ct.POINTER(ct.c_float)),
                ("d_R", ct.POINTER(ct.c_float)),
                ("d_Z", ct.POINTER(ct.c_int)),
                ("d_B", ct.POINTER(ct.c_float)),
                ("d_Gamma", ct.POINTER(ct.c_float)),
                ("d_GammaPrime", ct.POINTER(ct.c_float)),
                ("d_pi", ct.POINTER(ct.c_uint)),
                ("d_alphaBA", ct.POINTER(ct.c_float)),
                ]


_nova.pomdp_pbvi_complete_cpu.argtypes = (ct.POINTER(NovaPOMDP),
                                        ct.POINTER(ct.c_float), # Gamma
                                        ct.POINTER(ct.c_uint))  # pi

_nova.pomdp_pbvi_complete_gpu.argtypes = (ct.POINTER(NovaPOMDP),
                                        ct.c_uint,              # numThreadss
                                        ct.POINTER(ct.c_float), # Gamma
                                        ct.POINTER(ct.c_uint))  # pi


def pomdp_pbvi_complete_cpu(n, ns, m, z, r, rz, gamma, horizon,
        S, T, O, R, Z, B, numThreads, Gamma, pi):
    """ The wrapper Python function for executing point-based value iteration for a POMDP using the CPU.

        Parameters:
            n                   --  The number of states.
            ns                  --  The maximum number of successor states.
            m                   --  The number of actions.
            z                   --  The number of observations.
            r                   --  The number of belief points.
            rz                  --  The maximum number of non-zero belief values over all beliefs.
            gamma               --  The discount factor.
            horizon             --  The number of iterations.
            S                   --  The state-action pairs as a flattened 2-dimensional array.
            T                   --  The state transitions as a flattened 3-dimensional array.
            O                   --  The observation transitions as a flattened 3-dimensional array.
            R                   --  The reward function as a flattened 2-dimensional array.
            Z                   --  The belief-state pairs as a flattened 2-dimensional array.
            B                   --  The belief points as a flattened 2-dimensional array.
            Gamma               --  The resultant alpha-vectors. Modified.
            pi                  --  The resultant actions for each alpha-vector. Modified.

        Returns:
            Zero on success, and a non-zero nova error code otherwise.
    """

    global _nova

    array_type_nmns_int = ct.c_int * (int(n) * int(m) * int(ns))
    array_type_nmns_float = ct.c_float * (int(n) * int(m) * int(ns))
    array_type_mnz_float = ct.c_float * (int(m) * int(n) * int(z))
    array_type_nm_float = ct.c_float * (int(n) * int(m))
    array_type_rrz_int = ct.c_int * (int(r) * int(rz))
    array_type_rrz_float = ct.c_float * (int(r) * int(rz))

    array_type_rn_float = ct.c_float * (int(r) * int(n))
    array_type_r_uint = ct.c_uint * int(r)

    GammaResult = array_type_rn_float(*Gamma)
    piResult = array_type_r_uint(*pi)

    result = _nova.pomdp_pbvi_complete_cpu(NovaPOMDP(int(n), int(ns), int(m), int(z),
                                int(r), int(rz), float(gamma), int(horizon),
                                array_type_nmns_int(*S), array_type_nmns_float(*T),
                                array_type_mnz_float(*O), array_type_nm_float(*R),
                                array_type_rrz_int(*Z), array_type_rrz_float(*B),
                                int(0),
                                ct.POINTER(ct.c_float)(), ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_uint)(),
                                ct.POINTER(ct.c_int)(), ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_float)(), ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_int)(), ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_float)(), ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_uint)(),
                                ct.POINTER(ct.c_float)()),
                            GammaResult, piResult)

    if result == 0:
        for i in range(r * n):
            Gamma[i] = GammaResult[i]
        for i in range(r):
            pi[i] = piResult[i]

    return result


def pomdp_pbvi_complete_gpu(n, ns, m, z, r, rz, gamma, horizon,
        S, T, O, R, Z, B, numThreads, Gamma, pi):
    """ The wrapper Python function for executing point-based value iteration for a POMDP using the GPU.

        Parameters:
            n                   --  The number of states.
            ns                  --  The maximum number of successor states.
            m                   --  The number of actions.
            z                   --  The number of observations.
            r                   --  The number of belief points.
            rz                  --  The maximum number of non-zero belief values over all beliefs.
            gamma               --  The discount factor.
            horizon             --  The number of iterations.
            S                   --  The state-action pairs as a flattened 2-dimensional array.
            T                   --  The state transitions as a flattened 3-dimensional array.
            O                   --  The observation transitions as a flattened 3-dimensional array.
            R                   --  The reward function as a flattened 2-dimensional array.
            Z                   --  The belief-state pairs as a flattened 2-dimensional array.
            B                   --  The belief points as a flattened 2-dimensional array.
            numThreads          --  The number of CUDA threads to execute.
            Gamma               --  The resultant alpha-vectors. Modified.
            pi                  --  The resultant actions for each alpha-vector. Modified.

        Returns:
            Zero on success, and a non-zero nova error code otherwise.
    """

    global _nova

    array_type_nmns_int = ct.c_int * (int(n) * int(m) * int(ns))
    array_type_nmns_float = ct.c_float * (int(n) * int(m) * int(ns))
    array_type_mnz_float = ct.c_float * (int(m) * int(n) * int(z))
    array_type_nm_float = ct.c_float * (int(n) * int(m))
    array_type_rrz_int = ct.c_int * (int(r) * int(rz))
    array_type_rrz_float = ct.c_float * (int(r) * int(rz))

    array_type_rn_float = ct.c_float * (int(r) * int(n))
    array_type_r_uint = ct.c_uint * int(r)

    GammaResult = array_type_rn_float(*Gamma)
    piResult = array_type_r_uint(*pi)

    result = _nova.pomdp_pbvi_complete_gpu(NovaPOMDP(int(n), int(ns), int(m), int(z),
                                int(r), int(rz), float(gamma), int(horizon),
                                array_type_nmns_int(*S), array_type_nmns_float(*T),
                                array_type_mnz_float(*O), array_type_nm_float(*R),
                                array_type_rrz_int(*Z), array_type_rrz_float(*B),
                                int(0),
                                ct.POINTER(ct.c_float)(), ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_uint)(),
                                ct.POINTER(ct.c_int)(), ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_float)(), ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_int)(), ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_float)(), ct.POINTER(ct.c_float)(),
                                ct.POINTER(ct.c_uint)(),
                                ct.POINTER(ct.c_float)()),
                            int(numThreads), GammaResult, piResult)

    if result == 0:
        for i in range(r * n):
            Gamma[i] = GammaResult[i]
        for i in range(r):
            pi[i] = piResult[i]

    return result


class POMDP(NovaPOMDP):
    """ A Partially Observable Markov Decision Process (POMDP) object that can load, solve, and save.

        Specifically, it is capable of loading raw and cassandra-like POMDP files, provides
        functionality to solve them using the nova library, and enables saving the resulting
        policy as a raw policy file.
    """

    def __init__(self):
        """ The constructor for the POMDP class. """

        # Assign a nullptr for the device-side pointers. These will be set if the GPU is utilized.
        self.currentHorizon = int(0)
        self.Gamma = ct.POINTER(ct.c_float)()
        self.GammaPrime = ct.POINTER(ct.c_float)()
        self.pi = ct.POINTER(ct.c_uint)()
        self.d_S = ct.POINTER(ct.c_int)()
        self.d_T = ct.POINTER(ct.c_float)()
        self.d_O = ct.POINTER(ct.c_float)()
        self.d_R = ct.POINTER(ct.c_float)()
        self.d_Z = ct.POINTER(ct.c_int)()
        self.d_B = ct.POINTER(ct.c_float)()
        self.d_Gamma = ct.POINTER(ct.c_float)()
        self.d_GammaPrime = ct.POINTER(ct.c_float)()
        self.d_pi = ct.POINTER(ct.c_uint)()
        self.d_alphaBA = ct.POINTER(ct.c_float)()

        # Additional informative variables.
        self.Rmin = None

    def load(self, filename, scalarize=lambda x: x[0]):
        """ Load a raw Multi-Objective POMDP file given the filename and optionally a scalarization function.

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
            self.z = int(data[0][3])
            self.r = int(data[0][4])
            self.rz = int(data[0][5])
            self.k = int(data[0][6])

            self.horizon = int(data[0][7])
            self.gamma = float(data[0][8])

            # Functions to convert flattened NumPy arrays to C arrays.
            array_type_nmns_int = ct.c_int * (self.n * self.m * self.ns)
            array_type_nmns_float = ct.c_float * (self.n * self.m * self.ns)
            array_type_mnz_float = ct.c_float * (self.m * self.n * self.z)
            array_type_nm_float = ct.c_float * (self.n * self.m)
            array_type_rrz_int = ct.c_int * (self.r * self.rz)
            array_type_rrz_float = ct.c_float * (self.r * self.rz)

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
            self.O = array_type_mnz_float(*np.array([[[float(data[(self.z * a + o) + rowOffset][sp]) \
                                for o in range(self.z)] \
                            for sp in range(self.n)] \
                        for a in range(self.m)]).flatten())

            rowOffset = 1 + self.n * self.m + self.n * self.m + self.m * self.z
            self.R = array_type_nm_float(*scalarize(np.array([[[float(data[(self.m * i + a) + rowOffset][s])
                                for a in range(self.m)] \
                            for s in range(self.n)] \
                        for i in range(self.k)])).flatten())

            self.Rmin = min([self.R[i] for i in range(self.n * self.m)])

            rowOffset = 1 + self.n * self.m + self.n * self.m + self.m * self.z + self.k * self.m
            self.Z = array_type_rrz_int(*np.array([[int(data[i + rowOffset][s])
                            for s in range(self.rz)] \
                        for i in range(self.r)]).flatten())

            rowOffset = 1 + self.n * self.m + self.n * self.m + self.m * self.z + self.k * self.m + self.r
            self.B = array_type_rrz_float(*np.array([[float(data[i + rowOffset][s])
                            for s in range(self.rz)] \
                        for i in range(self.r)]).flatten())

        except Exception:
            print("Failed to load file '%s'." % (filename))
            raise Exception()

    def solve(self, numThreads=1024):
        """ Solve the POMDP using the nova Python wrapper.

            Parameters:
                numThreads  --  The number of CUDA threads to execute (multiple of 32). Default is 1024.

            Returns:
                Gamma   --  The alpha-vectors, one for each belief point, mapping states to values.
                pi      --  The policy, mapping alpha-vectors (belief points) to actions.
        """

        global _nova

        # Create Gamma and pi, assigning them their respective initial values.
        Gamma = np.array([[0.0 for s in range(self.n)] for b in range(self.r)])
        if self.gamma < 1.0:
            Gamma = np.array([[float(self.Rmin / (1.0 - self.gamma)) for s in range(self.n)] \
                        for i in range(self.r)])
        Gamma = Gamma.flatten()
        pi = np.array([0 for i in range(self.r)])

        # Create functions to convert flattened NumPy arrays to C arrays.
        array_type_rn_float = ct.c_float * (self.r * self.n)
        array_type_r_uint = ct.c_uint * self.r

        # Create C arrays for the result.
        GammaResult = array_type_rn_float(*Gamma)
        piResult = array_type_r_uint(*pi)

        result = _nova.pomdp_pbvi_complete_gpu(self, int(numThreads), GammaResult, piResult)
        if result != 0:
            result = _nova.pomdp_pbvi_complete_cpu(self, GammaResult, piResult)

        if result == 0:
            Gamma = np.array([GammaResult[i] for i in range(self.r * self.n)])
            pi = np.array([piResult[i] for i in range(self.r)])
        else:
            print("Failed to solve POMDP using the 'nova' library.")
            raise Exception()

        Gamma = Gamma.reshape((self.r, self.n))

        return Gamma, pi

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

        result += "S(s, a, s'):\n%s" % (str(np.array([self.S[i] \
                    for i in range(self.n * self.m * self.ns)]).reshape((self.n, self.m, self.ns)))) + "\n\n"

        result += "T(s, a, s'):\n%s" % (str(np.array([self.T[i] \
                    for i in range(self.n * self.m * self.ns)]).reshape((self.n, self.m, self.ns)))) + "\n\n"

        result += "O(a, s', o):\n%s" % (str(np.array([self.O[i] \
                    for i in range(self.m * self.n * self.z)]).reshape((self.m, self.n, self.z)))) + "\n\n"

        result += "R(s, a):\n%s" % (str(np.array([self.R[i] \
                    for i in range(self.n * self.m)]).reshape((self.n, self.m)))) + "\n\n"

        result += "Z(i, s):\n%s" % (str(np.array([self.Z[i] \
                    for i in range(self.r * self.rz)]).reshape((self.r, self.rz)))) + "\n\n"

        result += "B(i, s):\n%s" % (str(np.array([self.B[i] \
                    for i in range(self.r * self.rz)]).reshape((self.r, self.rz)))) + "\n\n"

        return result


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
import os
import sys
import csv
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))
import nova_mdp as nm


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

    def load(self, filename, filetype='mdp', scalarize=lambda x: x[0]):
        """ Load a raw Multi-Objective POMDP file given the filename and optionally the file type.

            Parameters:
                filename    --  The name and path of the file to load.
                filetype    --  Either 'mdp' or 'raw'. Default is 'mdp'.
                scalarize   --  Optionally define a scalarization function. Only used for 'raw' files.
                                Default returns the first reward.
        """

        if filetype == 'mdp':
            self._load_mdp(filename)
        elif filetype == 'raw':
            self._load_raw(filename, scalarize)
        else:
            print("Invalid file type '%s'." % (filetype))
            raise Exception()

    def _load_mdp(self, filename):
        """ Load a Cassandra-format MDP file given the filename and optionally a scalarization function.

            Parameters:
                filename    --  The name and path of the file to load.
                scalarize   --  Optionally define a scalarization function. Default returns the first reward.
        """

        # Attempt to parse all the data into their respective variables.
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    #row = str(row)
                    print(row)

        except Exception:
            print("Failed to load file '%s'." % (filename))
            raise Exception()

    def _load_raw(self, filename, scalarize=lambda x: x[0]):
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
            self.ng = int(0)
            self.goals = ct.POINTER(ct.c_uint)()

            self.horizon = int(data[0][5])
            self.gamma = float(data[0][6])
            self.epsilon = float(0.01)

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

            self.Rmax = max([self.R[i] for i in range(self.n * self.m)])
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

        # Create V and pi, assigning them their respective initial values.
        V = np.array([0.0 for s in range(self.n)])
        #if self.gamma < 1.0:
        #    V = np.array([float(self.Rmin / (1.0 - self.gamma)) for s in range(self.n)])
        pi = np.array([0 for s in range(self.n)])

        # Create functions to convert flattened NumPy arrays to C arrays.
        array_type_n_float = ct.c_float * self.n
        array_type_n_uint = ct.c_uint * self.n

        # Create C arrays for the result.
        VResult = array_type_n_float(*V)
        piResult = array_type_n_uint(*pi)

        # Solve the MDP using the nova library and return the solution. If it
        # fails to use the GPU version, then it tries the CPU version.
        result = nm._nova.mdp_vi_complete_gpu(self, int(numThreads), VResult, piResult)
        if result != 0:
            result = nm._nova.mdp_vi_complete_cpu(self, VResult, piResult)

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
        result += "goals:   " + str([self.goals[i] for i in range(self.k)]) + "\n"
        result += "horizon: " + str(self.horizon) + "\n"
        result += "gamma:   " + str(self.gamma) + "\n\n"

        result += "S(s, a, s'):\n%s" % (str(np.array([self.S[i] \
                    for i in range(self.n * self.m * self.ns)]).reshape((self.n, self.m, self.ns)))) + "\n\n"

        result += "T(s, a, s'):\n%s" % (str(np.array([self.T[i] \
                    for i in range(self.n * self.m * self.ns)]).reshape((self.n, self.m, self.ns)))) + "\n\n"

        result += "R(s, a):\n%s" % (str(np.array([self.R[i] \
                    for i in range(self.n * self.m)]).reshape((self.n, self.m)))) + "\n\n"

        return result


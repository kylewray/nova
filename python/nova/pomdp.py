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

import ctypes as ct
import os
import sys
import csv
import numpy as np
import time
import random

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))
import nova_pomdp as npm
import file_loader as fl

import nova_pomdp_alpha_vectors as npav
import pomdp_alpha_vectors as pav

csv.field_size_limit(sys.maxsize)


class POMDP(npm.NovaPOMDP):
    """ A Partially Observable Markov Decision Process (POMDP) object that can load, solve, and save.

        Specifically, it is capable of loading raw and cassandra-like POMDP files, provides
        functionality to solve them using the nova library, and enables saving the resulting
        policy as a raw policy file.
    """

    def __init__(self):
        """ The constructor for the POMDP class. """

        # Assign a nullptr for the device-side pointers. These will be set if the GPU is utilized.
        self.n = int(0)
        self.ns = int(0)
        self.m = int(0)
        self.z = int(0)
        self.r = int(0)
        self.rz = int(0)
        self.gamma = float(0.9)
        self.horizon = int(1)
        self.S = ct.POINTER(ct.c_int)()
        self.T = ct.POINTER(ct.c_float)()
        self.O = ct.POINTER(ct.c_float)()
        self.R = ct.POINTER(ct.c_float)()
        self.Z = ct.POINTER(ct.c_int)()
        self.B = ct.POINTER(ct.c_float)()
        self.d_S = ct.POINTER(ct.c_int)()
        self.d_T = ct.POINTER(ct.c_float)()
        self.d_O = ct.POINTER(ct.c_float)()
        self.d_R = ct.POINTER(ct.c_float)()
        self.d_Z = ct.POINTER(ct.c_int)()
        self.d_B = ct.POINTER(ct.c_float)()

        # Additional useful variables not in the structure.
        self.Rmin = None
        self.Rmax = None

        self.cpuIsInitialized = False
        self.gpuIsInitialized = False

    def __del__(self):
        """ The deconstructor for the POMDP class. """

        self.uninitialize_gpu()
        self.uninitialize()

    def __str__(self):
        """ Return the string of the POMDP values akin to the raw file format.

            Returns:
                The string of the MOPOMDP in a similar format as the raw file format.
        """

        result = "n:       " + str(self.n) + "\n"
        result += "ns:      " + str(self.ns) + "\n"
        result += "m:       " + str(self.m) + "\n"
        result += "z:       " + str(self.z) + "\n"
        result += "r:       " + str(self.r) + "\n"
        result += "rz:      " + str(self.rz) + "\n"
        result += "horizon: " + str(self.horizon) + "\n"
        result += "gamma:   " + str(self.gamma) + "\n"

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

    def initialize(self, n, ns, m, z, r, rz, gamma, horizon):
        """ Initialize the POMDP's internal arrays, allocating memory.

            Parameters:
                n       --  The number of states.
                ns      --  The maximum number of successors.
                m       --  The number of actions.
                z       --  The number of observations.
                r       --  The number of beliefs.
                rz      --  The maximum number of non-zero values in belief points.
                gamma   --  The discount factor between 0 and 1.
                horizon --  The positive value for the horizon.
        """

        if self.cpuIsInitialized:
            return

        result = npm._nova.pomdp_initialize(self, n, ns, m, z, r, rz, gamma, horizon)
        if result != 0:
            print("Failed to initialize the POMDP.")
            raise Exception()

        self.cpuIsInitialized = True

    def uninitialize(self):
        """ Uninitialize the POMDP's internal arrays, freeing memory. """

        if not self.cpuIsInitialized:
            return

        result = npm._nova.pomdp_uninitialize(self)
        if result != 0:
            print("Failed to uninitialize the POMDP.")
            raise Exception()

        self.cpuIsInitialized = False

    def initialize_gpu(self):
        """ Initialize the GPU variables. This only needs to be called if GPU algorithms are used. """

        if self.gpuIsInitialized:
            return

        result = npm._nova.pomdp_initialize_gpu(self)
        if result != 0:
            print("Failed to initialize the 'nova' library's GPU variables for the POMDP.")
            raise Exception()

        self.gpuIsInitialized = True

    def uninitialize_gpu(self):
        """ Uninitialize the GPU variables. This only needs to be called if GPU algorithms are used. """

        if not self.gpuIsInitialized:
            return

        result = npm._nova.pomdp_uninitialize_gpu(self)
        if result != 0:
            print("Failed to initialize the 'nova' library's GPU variables for the POMDP.")
            raise Exception()

        self.gpuIsInitialized = False

    def load(self, filename, filetype='cassandra', scalarize=lambda x: x[0]):
        """ Load a POMDP file given the filename and optionally the file type.

            Parameters:
                filename    --  The name and path of the file to load.
                filetype    --  Either 'cassandra' or 'raw'. Default is 'cassandra'.
                scalarize   --  Optionally define a scalarization function. Only used for 'raw' files.
                                Default returns the first reward.
        """

        # Before anything, uninitialize the current POMDP.
        self.uninitialize_gpu()
        self.uninitialize()

        # Now load the file based on the desired file type.
        fileLoader = fl.FileLoader()

        if filetype == 'cassandra':
            fileLoader.load_cassandra(filename)
        elif filetype == 'raw':
            fileLoader.load_raw_pomdp(filename, scalarize)
        else:
            print("Invalid file type '%s'." % (filetype))
            raise Exception()

        # Allocate the memory on the C-side. Note: Allocating on the Python-side will create managed pointers.
        self.initialize(fileLoader.n, fileLoader.ns, fileLoader.m,
                        fileLoader.z, fileLoader.r, fileLoader.rz,
                        fileLoader.gamma, fileLoader.horizon)

        # Flatten all the file loader data.
        fileLoader.S = fileLoader.S.flatten()
        fileLoader.T = fileLoader.T.flatten()
        fileLoader.O = fileLoader.O.flatten()
        fileLoader.R = fileLoader.R.flatten()
        fileLoader.Z = fileLoader.Z.flatten()
        fileLoader.B = fileLoader.B.flatten()

        # Copy all of the variables' data into these arrays.
        for i in range(self.n * self.m * self.ns):
            self.S[i] = fileLoader.S[i]
            self.T[i] = fileLoader.T[i]
        for i in range(self.m * self.n * self.z):
            self.O[i] = fileLoader.O[i]
        for i in range(self.n * self.m):
            self.R[i] = fileLoader.R[i]
        for i in range(self.r * self.rz):
            self.Z[i] = fileLoader.Z[i]
            self.B[i] = fileLoader.B[i]

        self.Rmin = fileLoader.Rmin
        self.Rmax = fileLoader.Rmax

    def expand(self, method='random', numBeliefsToAdd=1000, maxTrials=100, pemaPolicy=None, pemaAlgorithm=None):
        """ Expand the belief points by, for example, PBVI's original method, PEMA, or Perseus' random method.

            Parameters:
                method              --  The method to use for expanding belief points. Default is 'random'.
                                        Methods:
                                            'random'            Random trajectories through the belief space.
                                            'random_unique'     Random *unique* trajectories through the belief space.
                                            'distinct_beliefs'  Distinct belief point selection.
                                            'pema'              Point-based Error Minimization Algorithm (PEMA).
                numBeliefsToAdd     --  Optionally define the number of belief points to add. Used by the
                                        'random' and 'random_unique'. Default is 1000.
                maxTrials           --  Optionally define the maximum number of trials while searching for unique beliefs.
                                        Used by 'random_unique'. Default is 100.
                pemaPolicy          --  Optionally use any policy object for PEMA. Default is None.
                pemaAlgorithm       --  Optionally use any POMDP algorithm object for PEMA. Only used if
                                        the pemaPolicy is None. Default is None.
        """

        if method not in ["random", "random_unique", "distinct_beliefs", "pema"]:
            print("Failed to expand. Method '%s' is not defined." % (method))
            raise Exception()

        if method == "random":
            npm._nova.pomdp_expand_random(self, numBeliefsToAdd)
        elif method == "random_unique":
            npm._nova.pomdp_expand_random_unique(self, numBeliefsToAdd, maxTrials)
        elif method == "distinct_beliefs":
            npm._nova.pomdp_expand_distinct_beliefs(self)
        elif method == "pema":
            if pemaPolicy is None:
                pemaPolicy = pemaAlgorithm.solve()
            npm._nova.pomdp_expand_pema(self, ct.byref(pemaPolicy))

    def sigma_approximate(self, numDesiredNonZeroValues=1):
        """ Perform the sigma-approximation algorithm on the current set of beliefs.

            Parameters:
                numDesiredNonZeroValues     --  The desired maximal number of non-zero values in the beliefs.

            Returns:
                The sigma = min_{b in B} sigma_b.
        """

        sigma = ct.c_float(0.0)

        result = npm._nova.pomdp_sigma(self, numDesiredNonZeroValues, ct.byref(sigma))
        if result != 0:
            print("Failed to perform sigma-approximation.")
            raise Exception()

        return sigma.value

    def belief_update(self, b, a, o):
        """ Perform a belief update, given belief, action, and observation.

            Parameters:
                b   --  The current belief (numpy n-array).
                a   --  The action (index) taken.
                o   --  The resulting observation (index).

            Returns:
                The new belief (numpy n-array).
        """

        array_type_n_float = ct.c_float * (self.n)

        b = array_type_n_float(*b.flatten())
        a = int(a)
        o = int(o)

        #bp = array_type_n_float(*np.zeros(self.n).flatten())
        bp = ct.POINTER(ct.c_float)()

        result = npm._nova.pomdp_belief_update(self, b, a, o, ct.byref(bp))
        if result != 0:
            print("Failed to perform a belief update.")
            raise Exception()

        return np.array([bp[s] for s in range(self.n)])

    def random_successor(self, s, a):
        """ Return a random successor from the state transitions.

            Parameters:
                s   --  The state.
                a   --  The action.

            Returns:
                The random successor state.
        """

        successor = None
        current = 0.0
        target = random.random()

        for i in range(self.ns):
            sp = self.S[s * self.m * self.ns + a * self.ns + i]
            if sp < 0:
                break

            current += self.T[s * self.m * self.ns + a * self.ns + i]
            if current >= target:
                successor = sp
                break

        if successor is None:
            successor = random.choice([self.S[s * self.m * self.ns + a * self.ns + i]
                                       for i in range(self.ns)
                                       if self.S[s * self.m * self.ns + a * self.ns + i] >= 0])

        return successor

    def random_observation(self, a, sp):
        """ Return a random observation from the observation function.

            Parameters:
                a   --  The action.
                sp  --  The successor state.

            Returns:
                The random observation.
        """

        observation = None
        current = 0.0
        target = random.random()

        for o in range(self.z):
            current += self.O[a * self.n * self.z + sp * self.z + o]
            if current >= target:
                observation = o
                break

        if observation is None:
            observation = random.choice([o for o in range(self.z) if self.O[a * self.n * self.z + sp * self.z + o] > 0.0])

        return observation


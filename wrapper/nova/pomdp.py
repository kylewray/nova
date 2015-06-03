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

_nova.nova_pomdp_pbvi.argtypes = (ct.c_uint, # n
                                ct.c_uint, # m
                                ct.c_uint, # z
                                ct.c_uint, # r
                                ct.c_uint, # maxNonZeroBeliefs
                                ct.c_uint, # maxSuccessors
                                ct.POINTER(ct.c_float), # B
                                ct.POINTER(ct.c_float), # T
                                ct.POINTER(ct.c_float), # O
                                ct.POINTER(ct.c_float), # R
                                ct.POINTER(ct.c_bool), # available
                                ct.POINTER(ct.c_int), # nonZeroBeliefs
                                ct.POINTER(ct.c_int), # successors
                                ct.c_float, # gamma
                                ct.c_uint, # horizon
                                ct.c_uint, # numThreads
                                ct.POINTER(ct.c_float), # Gamma
                                ct.POINTER(ct.c_uint)) # pi

def nova_pomdp_pbvi(n, m, z, r, maxNonZeroBeliefs, maxSuccessors,
        B, T, O, R, available, nonZeroBeliefs, successors,
        gamma, horizon, numThreads, Gamma, pi):
    """ The wrapper Python function for executing point-based value iteration for a POMDP.

        Parameters:
            n                   --  The number of states.
            m                   --  The number of actions.
            z                   --  The number of observations.
            r                   --  The number of belief points.
            maxNonZeroBeliefs   --  The maximum number of non-zero belief values over all beliefs.
            maxSuccessors       --  The maximum number of successor states.
            B                   --  The belief points as a flattened 2-dimensional array.
            T                   --  The state transitions as a flattened 3-dimensional array.
            O                   --  The observation transitions as a flattened 3-dimensional array.
            R                   --  The reward function as a flattened 2-dimensional array.
            available           --  The belief-action pairs as a flattened 2-dimensional array.
            nonZeroBeliefs      --  The belief-state pairs as a flattened 2-dimensional array.
            successors          --  The state-action pairs as a flattened 2-dimensional array.
            gamma               --  The discount factor.
            horizon             --  The number of iterations.
            numThreads          --  The number of CUDA threads to execute.
            Gamma               --  The resultant alpha-vectors. Modified.
            pi                  --  The resultant actions for each alpha-vector. Modified.

        Returns:
            0   --  Success.
            1   --  Invalid arguments were passed.
            2   --  The number of blocks and threads is less than the number of states.
            3   --  A CUDA memcpy failed somewhere, which should also output to std::err.
    """

    global _nova

    array_type_rn_float = ct.c_float * (int(r) * int(n))
    array_type_nmn_float = ct.c_float * (int(n) * int(m) * int(n))
    array_type_mnz_float = ct.c_float * (int(m) * int(n) * int(z))
    array_type_nm_float = ct.c_float * (int(n) * int(m))
    array_type_rm_bool = ct.c_bool * (int(r) * int(m))
    array_type_rxb_int = ct.c_int * (int(r) * int(maxNonZeroBeliefs))
    array_type_nmxs_int = ct.c_int * (int(n) * int(m) * int(maxSuccessors))
    array_type_r_uint = ct.c_uint * int(r)

    GammaResult = array_type_rn_float(*Gamma)
    piResult = array_type_r_uint(*pi)

    result = _nova.nova_pomdp_pbvi(int(n), int(m), int(z), int(r), int(maxNonZeroBeliefs),
                            int(maxSuccessors), array_type_rn_float(*B),
                            array_type_nmn_float(*T), array_type_mnz_float(*O),
                            array_type_nm_float(*R), array_type_rm_bool(*available),
                            array_type_rxb_int(*nonZeroBeliefs), array_type_nmxs_int(*successors),
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

        self.n = 0 # The number of states.
        self.m = 0 # The number of actions.
        self.z = 0 # The number of observations.
        self.r = 0 # The number of belief points.
        self.k = 1 # The number of reward functions.

        self.T = list() # State transitions T(s, a, s').
        self.O = list() # Observation transitions T(a, s', o).
        self.R = list() # Rewards R_i(s, a).

        self.B = list() # The initial set of belief states.

        self.available = list() # At each belief point, the actions available there.

        self.maxNonZeroBeliefs = 0 # The maximum number of non-zero values in a belief point.
        self.nonZeroBeliefs = list() # The mapping of beliefs to non-zero index values.

        self.maxSuccessors = 0 # The maximum number of successors over all states.
        self.successors = list() # A mapping from states to actions to the list of successors.

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
            self.m = int(data[0][1])
            self.z = int(data[0][2])
            self.r = int(data[0][3])
            self.k = int(data[0][4])

            self.horizon = int(data[0][5])
            self.gamma = float(data[0][6])

            rowOffset = 1
            self.T = np.array([[[float(data[(self.n * a + s) + rowOffset][sp]) \
                                for sp in range(self.n)] \
                            for a in range(self.m)] \
                        for s in range(self.n)])

            rowOffset = 1 + self.n * self.m
            self.O = np.array([[[float(data[(self.z * a + o) + rowOffset][sp]) \
                                for o in range(self.z)] \
                            for sp in range(self.n)] \
                        for a in range(self.m)])

            rowOffset = 1 + self.n * self.m + self.m * self.z
            self.R = np.array([[[float(data[(self.m * i + a) + rowOffset][s])
                                for a in range(self.m)] \
                            for s in range(self.n)] \
                        for i in range(self.k)])

            rowOffset = 1 + self.n * self.m + self.m * self.z + self.k * self.m
            self.B = np.array([[float(data[i + rowOffset][s])
                            for s in range(self.n)] \
                        for i in range(self.r)])

            self._compute_optimization_variables()

        except Exception:
            print("Failed to load file '%s'." % (filename))
            raise Exception()

    def _compute_optimization_variables(self):
        """ Compute the 'available', 'nonZeroBeliefs', and 'successors' variables. """

        # First, available is an r-m array which maps beliefs to booleans if actions are available.
        self.available = np.array([[True for a in range(self.m)] for i in range(self.r)])

        # Second, nonZeroBeliefs is an r-maxNonZeroBeliefs array which states the indices of
        # the non-zero beliefs in the belief vector. We must compute the value maxNonZeroBeliefs.
        self.maxNonZeroBeliefs = max([len([i for i, bs in enumerate(b) if bs > 0.0]) \
                                        for b in self.B])

        # Now compute the actual values.
        self.nonZeroBeliefs = list()
        for b in self.B:
            nonZeroBelief = [i for i, bs in enumerate(b) if bs > 0.0]
            nonZeroBelief += [-1] * (self.maxNonZeroBeliefs - len(nonZeroBelief))
            self.nonZeroBeliefs += [nonZeroBelief]
        self.nonZeroBeliefs = np.array(self.nonZeroBeliefs)

        # Third, successors is an n-m-maxSuccessor array that contains the list of state indices
        # which have a non-zero state transition. We must compute the value of maxSuccessors.
        self.maxSuccessors = max([len([sp for sp in range(self.n) if self.T[s][a][sp] > 0.0]) \
                                for s, a in itertools.product(range(self.n), range(self.m))])

        # Now compute the actual values.
        self.successors = list()
        for s in range(self.n):
            succ = list()
            for a in range(self.m):
                successorsSA = [sp for sp in range(self.n) if self.T[s][a][sp] > 0.0]
                successorsSA += [-1] * (self.maxSuccessors - len(successorsSA))
                succ += [successorsSA]
            self.successors += [succ]
        self.successors = np.array(self.successors)

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
            result = nova_pomdp_pbvi(self.n, self.m, self.z, self.r,
                        self.maxNonZeroBeliefs, self.maxSuccessors,
                        self.B.flatten(), self.T.flatten(), self.O.flatten(), self.R[0].flatten(),
                        self.available.flatten(), self.nonZeroBeliefs.flatten(),
                        self.successors.flatten(), self.gamma, self.horizon, numThreads,
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
        result += "m:       " + str(self.m) + "\n"
        result += "z:       " + str(self.z) + "\n"
        result += "r:       " + str(self.r) + "\n"
        result += "k:       " + str(self.k) + "\n"
        result += "horizon: " + str(self.horizon) + "\n"
        result += "gamma:   " + str(self.gamma) + "\n\n"

        result += "T(s, a, s'):\n%s" % (str(np.array(self.T))) + "\n\n"
        result += "O(a, s', o):\n%s" % (str(np.array(self.O))) + "\n\n"

        for i in range(self.k):
            result += "R_%i(s, a):\n%s" % (i + 1, str(np.array(self.R[i]))) + "\n\n"

        result += "B:\n%s" % (str(np.array(self.B))) + "\n\n"

        result += "available:\n%s" % (str(np.array(self.available))) + "\n\n"
        result += "nonZeroBeliefs:\n%s" % (str(np.array(self.nonZeroBeliefs))) + "\n\n"
        result += "successors:\n%s" % (str(np.array(self.successors))) + "\n\n"

        return result


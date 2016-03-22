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
import time

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
        self.currentHorizon = int(0)
        self.BTilde = ct.POINTER(ct.c_uint)()
        self.Gamma = ct.POINTER(ct.c_float)()
        self.GammaPrime = ct.POINTER(ct.c_float)()
        self.pi = ct.POINTER(ct.c_uint)()
        self.piPrime = ct.POINTER(ct.c_uint)()
        self.d_S = ct.POINTER(ct.c_int)()
        self.d_T = ct.POINTER(ct.c_float)()
        self.d_O = ct.POINTER(ct.c_float)()
        self.d_R = ct.POINTER(ct.c_float)()
        self.d_Z = ct.POINTER(ct.c_int)()
        self.d_B = ct.POINTER(ct.c_float)()
        self.d_Gamma = ct.POINTER(ct.c_float)()
        self.d_GammaPrime = ct.POINTER(ct.c_float)()
        self.d_pi = ct.POINTER(ct.c_uint)()
        self.d_piPrime = ct.POINTER(ct.c_uint)()
        self.d_alphaBA = ct.POINTER(ct.c_float)()

        # Additional informative variables.
        self.Rmin = None
        self.Rmax = None
        self.epsilon = 0.01

    def load(self, filename, filetype='cassandra', scalarize=lambda x: x[0]):
        """ Load a POMDP file given the filename and optionally the file type.

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
            fileLoader.load_raw_pomdp(filename, scalarize)
        else:
            print("Invalid file type '%s'." % (filetype))
            raise Exception()

        self.n = fileLoader.n
        self.ns = fileLoader.ns
        self.m = fileLoader.m
        self.z = fileLoader.z
        self.r = fileLoader.r
        self.rz = fileLoader.rz

        self.gamma = fileLoader.gamma
        self.horizon = fileLoader.horizon
        self.epsilon = fileLoader.epsilon

        self.Rmin = fileLoader.Rmin
        self.Rmax = fileLoader.Rmax

        array_type_nmns_int = ct.c_int * (self.n * self.m * self.ns)
        array_type_nmns_float = ct.c_float * (self.n * self.m * self.ns)
        array_type_mnz_float = ct.c_float * (self.m * self.n * self.z)
        array_type_nm_float = ct.c_float * (self.n * self.m)
        array_type_rrz_int = ct.c_int * (self.r * self.rz)
        array_type_rrz_float = ct.c_float * (self.r * self.rz)

        self.S = array_type_nmns_int(*fileLoader.S.flatten())
        self.T = array_type_nmns_float(*fileLoader.T.flatten())
        self.O = array_type_mnz_float(*fileLoader.O.flatten())
        self.R = array_type_nm_float(*fileLoader.R.flatten())
        self.Z = array_type_rrz_int(*fileLoader.Z.flatten())
        self.B = array_type_rrz_float(*fileLoader.B.flatten())

    def expand(self, method='random', numBeliefsToAdd=1000, Gamma=None):
        """ Expand the belief points by, for example, PBVI's original method, PEMA, or Perseus' random method.

            Parameters:
                method              --  The method to use for expanding belief points. Default is 'random'.
                                        Methods:
                                            'random'            Random trajectories through the belief space.
                                            'distinct_beliefs'  Distinct belief point selection.
                                            'pema'              Point-based Error Minimization Algorithm (PEMA).
                numBeliefsToAdd     --  Optionally define the number of belief points to add. Used by the
                                        'random'. Default is 1000.
                Gamma               --  Optionally define the alpha-vectors of the soultion (r-n array). Used by the
                                        'pema' method. Default is None, which will automatically solve the POMDP.
        """

        if method not in ["random", "distinct_beliefs", "pema"]:
            print("Failed to expand. Method '%s' is not defined." % (method))
            raise Exception()

        # Non-random methods add different quantities of belief points.
        if method == "distinct_beliefs":
            numBeliefsToAdd = self.r
        elif method == "pema":
            numBeliefsToAdd = 1

        array_type_uint = ct.c_uint * (1)
        array_type_ndbpn_float = ct.c_float * (numBeliefsToAdd * self.n)

        maxNonZeroValues = array_type_uint(*np.array([0]))
        Bnew = array_type_ndbpn_float(*np.zeros(numBeliefsToAdd * self.n))

        if method == "random":
            npm._nova.pomdp_expand_random_cpu(self, numBeliefsToAdd, maxNonZeroValues, Bnew)
        elif method == "distinct_beliefs":
            npm._nova.pomdp_expand_distinct_beliefs_cpu(self, maxNonZeroValues, Bnew)
        elif method == "pema":
            if Gamma is None:
                policy, timings = self.solve()

            npm._nova.pomdp_expand_pema_cpu(self, ct.byref(policy), maxNonZeroValues, Bnew)

        # Reconstruct the compressed Z and B.
        rPrime = int(self.r + numBeliefsToAdd)
        rzPrime = max(self.rz, int(maxNonZeroValues[0]))

        array_type_rrz_int = ct.c_int * (rPrime * rzPrime)
        array_type_rrz_float = ct.c_float * (rPrime * rzPrime)

        ZPrime = array_type_rrz_int(*-np.ones(rPrime * rzPrime).astype(int))
        BPrime = array_type_rrz_float(*np.zeros(rPrime * rzPrime).astype(float))

        for i in range(self.r):
            for j in range(self.rz):
                ZPrime[i * rzPrime + j] = self.Z[i * self.rz + j]
                BPrime[i * rzPrime + j] = self.B[i * self.rz + j]

        for i in range(numBeliefsToAdd):
            j = 0
            for s in range(self.n):
                if Bnew[i * self.n + s] > 0.0:
                    ZPrime[(self.r + i) * rzPrime + j] = s
                    BPrime[(self.r + i) * rzPrime + j] = Bnew[i * self.n + s]
                    j += 1

        self.r = rPrime
        self.rz = rzPrime
        self.Z = ZPrime
        self.B = BPrime

    def sigma_approximate(self, rz=1):
        """ Perform the sigma-approximation algorithm on the current set of beliefs.

            Parameters:
                rz  --  The desired maximal number of non-zero values in the belief vectors.
            Returns:
                The sigma = min_{b in B} sigma_b.
        """

        array_type_rrz_float = ct.c_float * (self.r * rz)
        array_type_rrz_int = ct.c_int * (self.r * rz)

        array_type_1_float = ct.c_float * (1)

        Bnew = array_type_rrz_float(*np.zeros(self.r * rz).astype(float))
        Znew = array_type_rrz_int(*-np.ones(self.r * rz).astype(int))

        sigma = array_type_1_float(*np.array([0.0]).astype(float))

        result = npm._nova.pomdp_sigma_cpu(self, rz, Bnew, Znew, sigma)
        if result != 0:
            print("Failed to perform sigma-approximation.")
            raise Exception()

        self.rz = rz
        self.B = Bnew
        self.Z = Znew

        return sigma[0]

    def solve(self, algorithm='pbvi', process='gpu', numThreads=1024):
        """ Solve the POMDP using the nova Python wrapper.

            Parameters:
                algorithm   --  The method to use, either 'pbvi' or 'perseus'. Default is 'pbvi'.
                process     --  Use the 'cpu' or 'gpu'. If 'gpu' fails, it tries 'cpu'. Default is 'gpu'.
                numThreads  --  The number of CUDA threads to execute (multiple of 32). Default is 1024.

            Returns:
                policy  --  The alpha-vectors and associated actions, one for each belief point.
                timing  --  A pair (wall-time, cpu-time) for solver execution time, not including (un)initialization.
        """

        # Create Gamma and pi, assigning them their respective initial values.
        initialGamma = np.array([[0.0 for s in range(self.n)] for b in range(self.r)])
        if self.gamma < 1.0:
            initialGamma = np.array([[float(self.Rmin / (1.0 - self.gamma)) for s in range(self.n)] \
                        for i in range(self.r)])
        initialGamma = initialGamma.flatten()

        # Create a function to convert a flattened numpy arrays to a C array, then convert initialGamma.
        array_type_rn_float = ct.c_float * (self.r * self.n)
        initialGamma = array_type_rn_float(*initialGamma)

        policy = ct.POINTER(pav.POMDPAlphaVectors)()
        timing = None

        # If the process is 'gpu', then attempt to solve it. If an error arises, then
        # assign process to 'cpu' and attempt to solve it using that.
        if process == 'gpu':
            result = npm._nova.pomdp_initialize_successors_gpu(self)
            result += npm._nova.pomdp_initialize_state_transitions_gpu(self)
            result += npm._nova.pomdp_initialize_observation_transitions_gpu(self)
            result += npm._nova.pomdp_initialize_rewards_gpu(self)
            result += npm._nova.pomdp_initialize_nonzero_beliefs_gpu(self)
            result += npm._nova.pomdp_initialize_belief_points_gpu(self)
            if result != 0:
                print("Failed to initialize the POMDP variables for the 'nova' library's GPU POMDP solver.")
                process = 'cpu'

            timing = (time.time(), time.clock())

            if algorithm == 'pbvi':
                result = npm._nova.pomdp_pbvi_execute_gpu(self, int(numThreads), initialGamma, ct.byref(policy))
            #elif algorithm == 'perseus':
            #    result = npm._nova.pomdp_perseus_execute_gpu(self, int(numThreads), initialGamma, ct.byref(policy))
            else:
                print("Failed to solve the POMDP with the GPU using 'nova' because algorithm '%s' is undefined." % (algorithm))
                raise Exception()

            timing = (time.time() - timing[0], time.clock() - timing[1])

            if result != 0:
                print("Failed to execute the 'nova' library's GPU POMDP solver.")
                process = 'cpu'

            result = npm._nova.pomdp_uninitialize_successors_gpu(self)
            result += npm._nova.pomdp_uninitialize_state_transitions_gpu(self)
            result += npm._nova.pomdp_uninitialize_observation_transitions_gpu(self)
            result += npm._nova.pomdp_uninitialize_rewards_gpu(self)
            result += npm._nova.pomdp_uninitialize_nonzero_beliefs_gpu(self)
            result += npm._nova.pomdp_uninitialize_belief_points_gpu(self)
            if result != 0:
                # Note: Failing at uninitialization should not cause the CPU version to be executed.
                print("Failed to uninitialize the POMDP variables for the 'nova' library's GPU POMDP solver.")

        # If the process is 'cpu', then attempt to solve it.
        if process == 'cpu':
            timing = (time.time(), time.clock())

            if algorithm == 'pbvi':
                result = npm._nova.pomdp_pbvi_execute_cpu(self, initialGamma, ct.byref(policy))
            elif algorithm == 'perseus':
                result = npm._nova.pomdp_perseus_execute_cpu(self, initialGamma, ct.byref(policy))
            else:
                print("Failed to solve the POMDP with the GPU using 'nova' because algorithm '%s' is undefined." % (algorithm))
                raise Exception()

            timing = (time.time() - timing[0], time.clock() - timing[1])

            if result != 0:
                print("Failed to execute the 'nova' library's CPU POMDP solver.")
                raise Exception()

        # Dereference the pointer (this is how you do it in ctypes).
        policy = policy.contents

        return policy, timing

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

    def belief_update(self, b, a, o):
        """ Perform a belief update, given belief, action, and observation.

            Parameters:
                b   --  The current belief (numpy n-array).
                a   --  The action (index) taken.
                o   --  The resulting observation (index).

            Returns:
                The new belief (numpy n-array).
        """

        bp = np.zeros([0.0 for s in range(self.n)])


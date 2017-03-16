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

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))

import mdp
import mdp_value_function as mvf

import nova_mdp_vi as nmvi


class MDPValueIterationCPU(nmvi.NovaMDPValueIterationCPU):
    """ The value iteration solver for MDPs.

        This class provides a clean python wrapper for simple interactions with this solver.
    """

    def __init__(self, mdpObject, VInitial=None):
        """ The constructor for the MDPValueIterationCPU class.

            Parameters:
                mdpObject   --  The MDP object on which to run value iteration.
                VInitial    --  The initial values for each state. If undefined, then it is
                                zero for gamma = 1.0, and Rmin / (1 - gamma) otherwise.
        """

        self.mdp = mdpObject
        self.mdpPtr = ct.POINTER(mdp.MDP)(self.mdp)

        self.VInitial = VInitial
        if VInitial is None:
            if self.mdp.gamma < 1.0:
                VInitial = np.array([float(self.mdp.Rmin / (1.0 - self.mdp.gamma)) for s in range(self.mdp.n)])
            else:
                VInitial = np.array([0.0 for s in range(self.mdp.n)])

            array_type_n_float = ct.c_float * self.mdp.n
            self.VInitial = array_type_n_float(*VInitial)

        self.currentHorizon = int(0)
        self.V = ct.POINTER(ct.c_float)()
        self.VPrime = ct.POINTER(ct.c_float)()
        self.pi = ct.POINTER(ct.c_uint)()

        # Attempt to initialize the algorithm.
        result = nmvi._nova.mdp_vi_initialize_cpu(self.mdpPtr, self)
        if result != 0:
            print("Failed to initialize the value iteration (CPU) algorithm.")
            raise Exception()

    def __del__(self):
        """ The deconstructor for the MDPValueIterationCPU class which automatically frees memory. """

        result = nmvi._nova.mdp_vi_uninitialize_cpu(self.mdpPtr, self)
        if result != 0:
            print("Failed to free the value iteration (CPU) algorithm.")
            raise Exception()

    def solve(self):
        """ Solve the MDP by executing the solver.

            Returns:
                The MDPValueFunction policy solution to the MDP.
        """

        policy = mvf.MDPValueFunction()

        result = nmvi._nova.mdp_vi_execute_cpu(self.mdpPtr, self, policy)
        if result != 0:
            print("Failed to execute the 'nova' library's CPU MDP solver.")
            raise Exception()

        return policy

    def __str__(self):
        """ Return the string of the MDP value iteration.

            Returns:
                The string of the MDP value iteration.
        """

        result = "VInitial:\n%s" % (str(np.array([self.VInitial[i] \
                    for i in range(self.mdp.n)]))) + "\n\n"

        result += "currentHorizon: %i" % (self.currentHorizon) + "\n\n"

        result += "V:\n%s" % (str(np.array([self.V[i] \
                    for i in range(self.mdp.n)]))) + "\n\n"

        result += "VPrime:\n%s" % (str(np.array([self.VPrime[i] \
                    for i in range(self.mdp.n)]))) + "\n\n"

        result += "pi:\n%s" % (str(np.array([self.pi[i] \
                    for i in range(self.mdp.n)]))) + "\n\n"

        return result


class MDPValueIterationGPU(nmvi.NovaMDPValueIterationGPU):
    """ The value iteration solver for MDPs.

        This class provides a clean python wrapper for simple interactions with this solver.
    """

    def __init__(self, mdpObject, numThreads=1024, Vinitial=None):
        """ The constructor for the MDPValueIterationGPU class.

            Parameters:
                mdpObject   --  The MDP object on which to run value iteration.
                numThreads  --  The number of CUDA threads to execute (multiple of 32). Default is 1024.
                Vinitial    --  The initial values for each state. If undefined, then it is
                                zero for gamma = 1.0, and Rmin / (1 - gamma) otherwise.
        """

        self.mdp = mdpObject
        self.mdpPtr = ct.POINTER(mdp.MDP)(self.mdp)

        self.Vinitial = Vinitial
        if Vinitial is None:
            if self.mdp.gamma < 1.0:
                Vinitial = np.array([float(self.mdp.Rmin / (1.0 - self.mdp.gamma)) for s in range(self.mdp.n)])
            else:
                Vinitial = np.array([0.0 for s in range(self.mdp.n)])

            array_type_n_float = ct.c_float * self.mdp.n
            self.Vinitial = array_type_n_float(*Vinitial)

        self.currentHorizon = int(0)
        self.numThreads = numThreads
        self.d_V = ct.POINTER(ct.c_float)()
        self.d_VPrime = ct.POINTER(ct.c_float)()
        self.d_pi = ct.POINTER(ct.c_uint)()

        # Attempt to initialize the algorithm.
        result = nmvi._nova.mdp_vi_initialize_gpu(self.mdpPtr, self)
        if result != 0:
            print("Failed to initialize the value iteration (GPU) algorithm.")
            raise Exception()

    def __del__(self):
        """ The deconstructor for the MDPValueIterationGPU class which automatically frees memory. """

        result = nmvi._nova.mdp_vi_uninitialize_gpu(self.mdpPtr, self)
        if result != 0:
            print("Failed to free the value iteration (GPU) algorithm.")
            raise Exception()

    def solve(self):
        """ Solve the MDP by executing the solver.

            Returns:
                The MDPValueFunction policy solution to the MDP.
        """

        policy = mvf.MDPValueFunction()

        result = nmvi._nova.mdp_vi_execute_gpu(self.mdpPtr, self, policy)
        if result != 0:
            print("Failed to execute the 'nova' library's GPU MDP solver.")
            raise Exception()

        return policy

    def __str__(self):
        """ Return the string of the MDP value iteration.

            Returns:
                The string of the MDP value iteration.
        """

        result = "Vinitial:\n%s" % (str(np.array([self.Vinitial[i] \
                    for i in range(self.mdp.n)]))) + "\n\n"

        result += "currentHorizon: %i" % (self.currentHorizon) + "\n\n"
        result += "numThreads: %i" % (self.numThreads) + "\n\n"

        return result



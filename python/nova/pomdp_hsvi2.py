""" The MIT License (MIT)

    Copyright (c) 2017 Kyle Hollins Wray, University of Massachusetts

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

import pomdp
import pomdp_alpha_vectors as pav

import nova_pomdp_hsvi2 as nph


class POMDPHSVI2CPU(nph.NovaPOMDPHSVI2CPU):
    """ The point-based value iteration solver for POMDPs.

        This class provides a clean python wrapper for simple interactions with this solver.
    """

    def __init__(self, pomdpObject):
        """ The constructor for the POMDPHSVI2CPU class.

            Parameters:
                pomdpObject     --  The POMDP object on which to run HSVI2.
        """

        self.pomdp = pomdpObject
        self.pomdpPtr = ct.POINTER(pomdp.POMDP)(self.pomdp)

        self.trials = int(1)
        self.epsilon = float(0.01)
        self.pruneGrowthThreshold = float(0.1)
        self.maxAlphaVectors = int(max(self.pomdp.n, self.pomdp.m) + self.trials * self.pomdp.horizon + 1)
        self.currentTrial = int(0)
        self.lowerGammaSize = int(0)
        self.lowerGammaSizeLastPruned = int(0)
        self.lowerGamma = ct.POINTER(ct.c_float)()
        self.lowerPi = ct.POINTER(ct.c_uint)()
        self.upperGammaSize = int(0)
        self.upperGammaSizeLastPruned = int(0)
        self.upperGammaB = ct.POINTER(ct.c_float)()
        self.upperGammaHVb = ct.POINTER(ct.c_float)()

        # Attempt to initialize the algorithm.
        result = nph._nova.pomdp_hsvi2_initialize_cpu(self.pomdpPtr, self)
        if result != 0:
            print("Failed to initialize the HSVI2 (CPU) algorithm.")
            raise Exception()

    def __del__(self):
        """ The deconstructor for the POMDPHSVI2CPU class which automatically frees memory. """

        result = nph._nova.pomdp_hsvi2_uninitialize_cpu(self.pomdpPtr, self)
        if result != 0:
            print("Failed to free the HSVI2 (CPU) algorithm.")
            raise Exception()

    def solve(self):
        """ Solve the POMDP by executing the solver.

            Returns:
                The POMDPAlphaVectors policy solution to the POMDP.
        """

        policy = pav.POMDPAlphaVectors()

        result = nph._nova.pomdp_hsvi2_execute_cpu(self.pomdpPtr, self, policy)
        if result != 0:
            print("Failed to execute the 'nova' library's CPU POMDP solver.")
            raise Exception()

        return policy

    def update(self):
        """ Update the POMDP by executing one step of the solver. """

        result = nph._nova.pomdp_hsvi2_update_cpu(self.pomdpPtr, self)
        if result != 0:
            print("Failed to update the 'nova' library's CPU POMDP solver.")
            raise Exception()

    def get_policy(self):
        """ Get the policy computed by the solver.

            Returns:
                The POMDPAlphaVectors policy solution to the POMDP.
        """

        policy = pav.POMDPAlphaVectors()

        result = nph._nova.pomdp_hsvi2_get_policy_cpu(self.pomdpPtr, self, policy)
        if result != 0:
            print("Failed to get the policy for the 'nova' library's CPU POMDP solver.")
            raise Exception()

        return policy

    def __str__(self):
        """ Return the string of the POMDP HSVI2.

            Returns:
                The string of the POMDP HSVI2.
        """

        result = "trials: %i" % (self.trials) + "\n\n"
        result += "epsilon: %.3f" % (self.epsilon) + "\n\n"
        result += "pruneGrowthThreshold: %.3f" % (self.pruneGrowthThreshold) + "\n\n"
        result += "maxAlphaVectors: %i" % (self.maxAlphaVectors) + "\n\n"

        result += "currentTrial: %i" % (self.currentTrial) + "\n\n"

        result += "lowerGammaSize: %i" % (self.lowerGammaSize) + "\n\n"
        result += "lowerGamma:\n%s" % (str(np.array([[self.lowerGamma[j * self.pomdp.n + i] \
                        for j in range(self.lowerGammaSize)] \
                    for i in range(self.pomdp.n)]))) + "\n\n"
        result += "lowerPi:\n%s" % (str(np.array([self.lowerPi[i] \
                    for i in range(self.lowerGammaSize)]))) + "\n\n"

        result += "upperGammaSize: %i" % (self.upperGammaSize) + "\n\n"
        result += "upperGammaB:\n%s" % (str(np.array([[self.upperGammaB[j * self.pomdp.n + i] \
                        for j in range(self.upperGammaSize)] \
                    for i in range(self.pomdp.n)]))) + "\n\n"
        result += "upperGammaHVb:\n%s" % (str(np.array([self.upperGammaHVb[i] \
                    for i in range(self.lowerGammaSize)]))) + "\n\n"

        return result



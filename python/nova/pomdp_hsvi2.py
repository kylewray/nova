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

    def __init__(self, pomdpObject, GammaInitial=None):
        """ The constructor for the POMDPHSVI2CPU class.

            Parameters:
                pomdpObject     --  The POMDP object on which to run HSVI2.
                GammaInitial    --  The initial values for each belief state. If undefined, then it is
                                    zero for gamma = 1.0, and Rmin / (1 - gamma) otherwise.
        """

        self.pomdp = pomdpObject
        self.pomdpPtr = ct.POINTER(pomdp.POMDP)(self.pomdp)

        self.GammaInitial = GammaInitial
        if GammaInitial is None:
            if self.pomdp.gamma < 1.0:
                GammaInitial = np.array([[float(self.pomdp.Rmin / (1.0 - self.pomdp.gamma))
                                        for s in range(self.pomdp.n)] \
                                    for i in range(self.pomdp.r)])
            else:
                GammaInitial = np.array([[0.0 for s in range(self.pomdp.n)] for b in range(self.pomdp.r)])
            GammaInitial = GammaInitial.flatten()

            array_type_rn_float = ct.c_float * (self.pomdp.r * self.pomdp.n)
            self.GammaInitial = array_type_rn_float(*GammaInitial)

        self.currentHorizon = int(0)
        self.Gamma = ct.POINTER(ct.c_float)()
        self.GammaPrime = ct.POINTER(ct.c_float)()
        self.pi = ct.POINTER(ct.c_uint)()

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

        result = "GammaInitial:\n%s" % (str(np.array([[self.GammaInitial[j * self.pomdp.n + i] \
                        for j in range(self.pomdp.r)] \
                    for i in range(self.pomdp.n)]))) + "\n\n"

        result += "currentHorizon: %i" % (self.currentHorizon) + "\n\n"

        result += "Gamma:\n%s" % (str(np.array([[self.Gamma[j * self.pomdp.n + i] \
                        for j in range(self.pomdp.r)] \
                    for i in range(self.pomdp.n)]))) + "\n\n"

        result += "GammaPrime:\n%s" % (str(np.array([[self.GammaPrime[j * self.pomdp.n + i] \
                        for j in range(self.pomdp.r)] \
                    for i in range(self.pomdp.n)]))) + "\n\n"

        result += "pi:\n%s" % (str(np.array([self.pi[i] \
                    for i in range(self.pomdp.n)]))) + "\n\n"

        return result



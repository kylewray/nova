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

import pomdp
import pomdp_alpha_vectors as pav

import nova_pomdp_perseus as npper


class POMDPPerseusCPU(npper.NovaPOMDPPerseusCPU):
    """ The Perseus solver for POMDPs.

        This class provides a clean python wrapper for simple interactions with this solver.
    """

    def __init__(self, pomdpObject, GammaInitial=None):
        """ The constructor for the POMDPPerseusCPU class.

            Parameters:
                pomdpObject     --  The POMDP object on which to run Perseus.
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
        self.rGamma = int(0)
        self.rGammaPrime = int(0)
        self.rTilde = int(0)
        self.BTilde = ct.POINTER(ct.c_uint)()
        self.Gamma = ct.POINTER(ct.c_float)()
        self.GammaPrime = ct.POINTER(ct.c_float)()
        self.pi = ct.POINTER(ct.c_uint)()
        self.piPrime = ct.POINTER(ct.c_uint)()

        # Attempt to initialize the algorithm.
        result = npper._nova.pomdp_perseus_initialize_cpu(self.pomdpPtr, self)
        if result != 0:
            print("Failed to initialize the Perseus (CPU) algorithm.")
            raise Exception()

    def __del__(self):
        """ The deconstructor for the POMDPPerseusCPU class which automatically frees memory. """

        result = npper._nova.pomdp_perseus_uninitialize_cpu(self.pomdpPtr, self)
        if result != 0:
            print("Failed to free the Perseus (CPU) algorithm.")
            raise Exception()

    def solve(self):
        """ Solve the POMDP by executing the solver.

            Returns:
                The POMDPAlphaVectors policy solution to the POMDP.
        """

        policy = pav.POMDPAlphaVectors()

        result = npper._nova.pomdp_perseus_execute_cpu(self.pomdpPtr, self, policy)
        if result != 0:
            print("Failed to execute the 'nova' library's CPU POMDP solver.")
            raise Exception()

        return policy

    def __str__(self):
        """ Return the string of the POMDP Perseus.

            Returns:
                The string of the POMDP Perseus.
        """

        result = "GammaInitial:\n%s" % (str(np.array([[self.GammaInitial[j * self.pomdp.n + i] \
                        for j in range(self.pomdp.r)] \
                    for i in range(self.pomdp.n)]))) + "\n\n"

        result += "currentHorizon: %i" % (self.currentHorizon) + "\n\n"
        result += "rGamma: %i" % (self.rGamma) + "\n\n"
        result += "rGammaPrime: %i" % (self.rGammaPrime) + "\n\n"

        result += "rTilde: %i" % (self.rTilde) + "\n\n"
        result += "BTilde:\n%s" % (str(np.array([self.BTilde[i] for i in range(self.rTilde)]))) + "\n\n"

        result += "Gamma:\n%s" % (str(np.array([[self.Gamma[j * self.pomdp.n + i] \
                        for j in range(self.pomdp.r)] \
                    for i in range(self.pomdp.n)]))) + "\n\n"

        result += "GammaPrime:\n%s" % (str(np.array([[self.GammaPrime[j * self.pomdp.n + i] \
                        for j in range(self.pomdp.r)] \
                    for i in range(self.pomdp.n)]))) + "\n\n"

        result += "pi:\n%s" % (str(np.array([self.pi[i] \
                    for i in range(self.pomdp.r)]))) + "\n\n"

        result += "piPrime:\n%s" % (str(np.array([self.piPrime[i] \
                    for i in range(self.pomdp.r)]))) + "\n\n"

        return result


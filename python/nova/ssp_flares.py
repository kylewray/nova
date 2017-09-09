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

import mdp
import mdp_value_function as mvf

import nova_ssp_flares as nsl


class SSPFlaresCPU(nsl.NovaSSPFlaresCPU):
    """ The Flares solver for SSP MDPs.

        This class provides a clean python wrapper for simple interactions with this solver.
    """

    def __init__(self, mdpObject, VInitial=None):
        """ The constructor for the SSPFlaresCPU class.

            Parameters:
                mdpObject   --  The MDP object on which to run value iteration.
                VInitial    --  The heuristic values for all states. If undefined, then it is zero.
        """

        self.mdp = mdpObject
        self.mdpPtr = ct.POINTER(mdp.MDP)(self.mdp)

        self.VInitial = VInitial
        if VInitial is None:
            VInitial = np.array([0.0 for s in range(self.mdp.n)])
            array_type_n_float = ct.c_float * self.mdp.n
            self.VInitial = array_type_n_float(*VInitial)

        self.trials = int(1)
        self.currentTrial = int(0)
        self.currentHorizon = int(0)
        self.V = ct.POINTER(ct.c_float)()
        self.pi = ct.POINTER(ct.c_uint)()

        # Attempt to initialize the algorithm.
        result = nsl._nova.ssp_flares_initialize_cpu(self.mdpPtr, self)
        if result != 0:
            print("Failed to initialize the Flares (CPU) algorithm.")
            raise Exception()

    def __del__(self):
        """ The deconstructor for the SSPFlaresCPU class which automatically frees memory. """

        result = nsl._nova.ssp_flares_uninitialize_cpu(self.mdpPtr, self)
        if result != 0:
            print("Failed to free the Flares (CPU) algorithm.")
            raise Exception()

    def solve(self):
        """ Solve the SSP MDP by executing the solver.

            Returns:
                The MDPValueFunction policy solution to the SSP MDP.
        """

        policy = mvf.MDPValueFunction()

        result = nsl._nova.ssp_flares_execute_cpu(self.mdpPtr, self, ct.byref(policy))
        if result != 0 and result != 13: # Success or approximate solution.
            print("Failed to execute the 'nova' library's CPU Flares solver.")
            raise Exception()

        return policy

    def __str__(self):
        """ Return the string of the SSP Flares algorithm.

            Returns:
                The string of the SSP Flares algorithm.
        """

        result = "VInitial:\n%s" % (str(np.array([self.VInitial[i] \
                    for i in range(self.mdp.n)]))) + "\n\n"

        result += "trials: %i" % (self.trials) + "\n\n"

        result += "currentTrial: %i" % (self.currentTrial) + "\n\n"

        result += "currentHorizon: %i" % (self.currentHorizon) + "\n\n"

        result += "V:\n%s" % (str(np.array([self.V[i] \
                    for i in range(self.mdp.n)]))) + "\n\n"

        result += "pi:\n%s" % (str(np.array([self.pi[i] \
                    for i in range(self.mdp.n)]))) + "\n\n"

        return result



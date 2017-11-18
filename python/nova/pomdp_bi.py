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
import pomdp_stochastic_fsc as psfsc

import nova_pomdp_bi as npbi


class POMDPBeliefInfusion(npbi.NovaPOMDPBeliefInfusion):
    """ The belief-infused non-linear programming (NLP) solver for POMDPs.

        This class provides a clean python wrapper for simple interactions with this solver.
    """

    def __init__(self, pomdpObject, path, command, k, r):
        """ The constructor for the POMDPBeliefInfusion class.

            Parameters:
                pomdpObject     --  The POMDP object on which to run belief-infused NLP.
                path            --  The path to the folder to store temporary AMPL files.
                command         --  The command to use the generated AMPL files in a solver.
                k               --  The number of controller nodes.
                r               --  The number of beliefs to infuse into the FSC.
        """

        self.pomdp = pomdpObject
        self.pomdpPtr = ct.POINTER(pomdp.POMDP)(self.pomdp)

        self.path = ct.create_string_buffer(str.encode(path))
        self.command = ct.create_string_buffer(str.encode(command))
        self.k = int(k)
        self.r = int(r)
        self.B = ct.POINTER(ct.c_float)()
        self.lmbd = ct.POINTER(ct.c_float)()
        self.policy = ct.POINTER(ct.c_float)()

        # Attempt to initialize the algorithm.
        result = npbi._nova.pomdp_bi_initialize(self.pomdpPtr, self)
        if result != 0:
            print("Failed to initialize the belief-infused NLP algorithm.")
            raise Exception()

    def __del__(self):
        """ The deconstructor for the POMDPBeliefInfusion class which automatically frees memory. """

        result = npbi._nova.pomdp_bi_uninitialize(self.pomdpPtr, self)
        if result != 0:
            print("Failed to free the belief-infused NLP algorithm.")
            raise Exception()

    def __str__(self):
        """ Return the string of the POMDP belief-infused NLP.

            Returns:
                The string of the POMDP belief-infused NLP.
        """


        result = "path: %s" % (self.path) + "\n"
        result += "command: %s" % (self.command) + "\n"
        result += "k: %i" % (self.k) + "\n"
        result += "r: %i" % (self.k) + "\n\n"

        result += "B:\n%s" % (str(np.array([[self.B[i * self.pomdp.n + s] \
                        for s in range(self.pomdp.n)] \
                    for i in range(self.r)]))) + "\n\n"

        result += "lambda:\n%s" % (str(np.array([[self.lmbd[q * self.r + i] \
                        for i in range(self.r)] \
                    for q in range(self.k)]))) + "\n\n"

        result += "policy:\n%s" % (str(np.array([[[[self.policy[q * self.pomdp.n * self.pomdp.z * self.k
                                                                + s * self.pomdp.z * self.k
                                                                + o * self.k + qp] \
                                        for qp in range(self.k)] \
                                    for o in range(self.pomdp.z)] \
                                for s in range(self.pomdp.n)] \
                            for q in range(self.k)]))) + "\n\n"

        return result

    def solve(self):
        """ Solve the POMDP by executing the solver.

            Returns:
                The POMDPStochasticFSC policy solution to the POMDP.
        """

        policy = psfsc.POMDPStochasticFSC()

        result = npbi._nova.pomdp_bi_execute(self.pomdpPtr, self, policy)
        if result != 0:
            print("Failed to execute the 'nova' library's CPU POMDP solver.")
            raise Exception()

        return policy

    def update(self):
        """ Update the POMDP by executing one step of the solver. """

        result = npbi._nova.pomdp_bi_update(self.pomdpPtr, self)
        if result != 0:
            print("Failed to update the 'nova' library's CPU POMDP solver.")
            raise Exception()

    def get_policy(self):
        """ Get the policy computed by the solver.

            Returns:
                The POMDPStochasticFSC policy solution to the POMDP.
        """

        policy = psfsc.POMDPStochasticFSC()

        result = npbi._nova.pomdp_bi_get_policy(self.pomdpPtr, self, policy)
        if result != 0:
            print("Failed to get the policy for the 'nova' library's CPU POMDP solver.")
            raise Exception()

        return policy




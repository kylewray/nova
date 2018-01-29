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

import nova_pomdp_cbnlp as npcb


class POMDPCBNLP(npcb.NovaPOMDPCBNLP):
    """ The compressed-belief non-linear programming (NLP) solver for POMDPs.

        This class provides a clean python wrapper for simple interactions with this solver.
    """

    def __init__(self, pomdpObject, path, command, k, r, lmbd=0.1):
        """ The constructor for the POMDPCBNLP class.

            Parameters:
                pomdpObject     --  The POMDP object on which to run compressed-belief NLP.
                path            --  The path to the folder to store temporary AMPL files.
                command         --  The command to use the generated AMPL files in a solver.
                k               --  The number of controller nodes.
                r               --  The number of beliefs to infuse into the FSC.
                lmbd            --  The lambda in [0, 1] parameter; i.e., how much PBVI weight.
        """

        self.pomdp = pomdpObject
        self.pomdpPtr = ct.POINTER(pomdp.POMDP)(self.pomdp)

        self.path = ct.create_string_buffer(str.encode(path))
        self.command = ct.create_string_buffer(str.encode(command))
        self.k = int(k)
        self.r = int(r)
        self.numTotalNodes = int(k) + int(r)
        self.B = ct.POINTER(ct.c_float)()
        self.lmbd = float(lmbd)
        self.psi = ct.POINTER(ct.c_float)()
        self.eta = ct.POINTER(ct.c_float)()
        self.V = ct.POINTER(ct.c_float)()

        # Attempt to initialize the algorithm.
        result = npcb._nova.pomdp_cbnlp_initialize(self.pomdpPtr, self)
        if result != 0:
            print("Failed to initialize the compressed-belief NLP algorithm.")
            raise Exception()

    def __del__(self):
        """ The deconstructor for the POMDPCBNLP class which automatically frees memory. """

        result = npcb._nova.pomdp_cbnlp_uninitialize(self.pomdpPtr, self)
        if result != 0:
            print("Failed to free the compressed-belief NLP algorithm.")
            raise Exception()

    def __str__(self):
        """ Return the string of the POMDP compressed-belief NLP.

            Returns:
                The string of the POMDP compressed-belief NLP.
        """

        result = "path: %s" % (self.path) + "\n"
        result += "command: %s" % (self.command) + "\n"
        result += "k: %i" % (self.k) + "\n"
        result += "r: %i" % (self.r) + "\n\n"

        result += "B:\n%s" % (str(np.array([[self.B[i * self.pomdp.n + s] \
                                for s in range(self.pomdp.n)] \
                            for i in range(self.r)]))) + "\n\n"

        result += "lambda: %s" % (self.lmbd) + "\n\n"

        result += "psi:\n%s" % (str(np.array([[self.psi[i * self.pomdp.m + a] \
                                for a in range(self.pomdp.m)] \
                            for i in range(self.numTotalNodes)]))) + "\n\n"

        result += "eta:\n%s" % (str(np.array([[[[self.eta[x * self.pomdp.n * self.pomdp.z * self.numTotalNodes
                                                          + s * self.pomdp.z * self.numTotalNodes
                                                          + o * self.numTotalNodes + xp] \
                                        for xp in range(self.numTotalNodes)] \
                                    for o in range(self.pomdp.z)] \
                                for s in range(self.pomdp.n)] \
                            for x in range(self.numTotalNodes)]))) + "\n\n"

        result += "V:\n%s" % (str(np.array([[self.V[i * self.pomdp.n + s] \
                                for s in range(self.pomdp.n)] \
                            for i in range(self.numTotalNodes)]))) + "\n\n"

        return result

    def solve(self):
        """ Solve the POMDP by executing the solver.

            Returns:
                The POMDPStochasticFSC policy solution to the POMDP.
        """

        policy = psfsc.POMDPStochasticFSC()

        result = npcb._nova.pomdp_cbnlp_execute(self.pomdpPtr, self, policy)
        if result != 0:
            print("Failed to execute the 'nova' library's CPU POMDP solver.")
            raise Exception()

        return policy

    def update(self):
        """ Update the POMDP by executing one step of the solver. """

        result = npcb._nova.pomdp_cbnlp_update(self.pomdpPtr, self)
        if result != 0:
            print("Failed to update the 'nova' library's CPU POMDP solver.")
            raise Exception()

    def get_policy(self):
        """ Get the policy computed by the solver.

            Returns:
                The POMDPStochasticFSC policy solution to the POMDP.
        """

        policy = psfsc.POMDPStochasticFSC()

        result = npcb._nova.pomdp_cbnlp_get_policy(self.pomdpPtr, self, policy)
        if result != 0:
            print("Failed to get the policy for the 'nova' library's CPU POMDP solver.")
            raise Exception()

        return policy




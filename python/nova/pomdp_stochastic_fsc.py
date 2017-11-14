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
import nova_pomdp_stochastic_fsc as npfsc


class POMDPStochasticFSC(npfsc.NovaPOMDPStochasticFSC):
    """ The stochastic FSC representation of a POMDP policy.

        Specifically, this class is a clean python wrapper around using stochastic FSCs,
        as well as freeing the memory once created.
    """

    def __init__(self):
        """ The constructor for the POMDPStochasticFSC class. """

        self.k = 0
        self.m = 0
        self.z = 0
        self.psi = ct.POINTER(ct.c_float)()
        self.eta = ct.POINTER(ct.c_float)()

    def __del__(self):
        """ Free the memory of the policy when this object is deleted. """

        result = npfsc._nova.pomdp_stochastic_fsc_uninitialize(self)
        if result != 0:
            print("Failed to free the stochastic FSC.")
            raise Exception()

    def __str__(self):
        """ Return the string of the POMDP stochastic FSC.

            Returns:
                The string of the POMDP stochastic FSC.
        """

        result = "k: %i" % (self.k) + "\n\n"
        result = "m: %i" % (self.m) + "\n\n"
        result = "z: %i" % (self.z) + "\n\n"

        result += "psi:\n%s" % (str(np.array([[self.psi[i * self.n + a] \
                                            for i in range(self.k)] \
                                        for a in range(self.m)]))) + "\n\n"

        result += "eta:\n%s" % (str(np.array([[self.eta[i * self.k + qp] \
                                    for i in range(self.k * self.m * self.z)] \
                                for qp in range(self.k)]))) + "\n\n"

        return result

    def random_action(self, q):
        """ Take a random action following psi given the controller node q.

            Parameters:
                q   --  The current controller node.

            Returns:
                A random action selected following psi.
        """

        q = ct.c_uint(q)
        a = ct.c_uint(0)

        result = npfsc._nova.pomdp_stochastic_fsc_random_action(self, q, ct.byref(a))
        if result != 0:
            print("Failed to select a random action.")
            raise Exception()

        return a.value

    def random_successor(self, q, a, o):
        """ Select a random successor following eta given the node, action, and observation.

            Parameters:
                q   --  The current controller node.
                a   --  The action taken at q.
                o   --  The observate made after the action was taken.

            Returns:
                A random successor selected following eta.
        """

        q = ct.c_uint(q)
        a = ct.c_uint(a)
        o = ct.c_uint(o)
        qp = ct.c_uint(0)

        result = npfsc._nova.pomdp_stochastic_fsc_random_successor(self, q, a, o, ct.byref(qp))
        if result != 0:
            print("Failed to select a random successor.")
            raise Exception()

        return qp.value

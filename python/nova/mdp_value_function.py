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
import nova_mdp_value_function as nmvf


class MDPValueFunction(nmvf.NovaMDPValueFunction):
    """ The value function representation of an MDP policy.

        Specifically, this class is a clean python wrapper around simple operations,
        such as freeing the memory.
    """

    def __init__(self):
        """ The constructor for the MDPValueFunction class. """

        self.n = 0
        self.m = 0
        self.r = 0
        self.S = ct.POINTER(ct.c_uint)()
        self.V = ct.POINTER(ct.c_float)()
        self.pi = ct.POINTER(ct.c_uint)()

    def __del__(self):
        """ Free the memory of the policy when this object is deleted. """

        result = nmvf._nova.mdp_value_function_free(self)
        if result != 0:
            print("Failed to free the value function.")
            raise Exception()

    def __str__(self):
        """ Return the string of the MDP value function.

            Returns:
                The string of the MDP value function.
        """

        result = "S:\n%s" % (str(np.array([self.S[i] \
                    for i in range(self.r)]))) + "\n\n"

        result += "V:\n%s" % (str(np.array([[self.Gamma[i * self.n + s] \
                    for i in range(self.r)] for s in range(self.n)]))) + "\n\n"

        result += "pi:\n%s" % (str(np.array([self.pi[i] \
                    for i in range(self.r)]))) + "\n\n"

        return result


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
import random

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

        result = nmvf._nova.mdp_value_function_uninitialize(self)
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

    def value_and_action(self, s):
        """ Compute the optimal value and action at a state.

            Parameters:
                s   --  The state (index).

            Returns:
                V   --  The optimal value at this state.
                a   --  The optimal action at this state.
        """

        Vs = self.V[s].value
        a = self.pi[s].value

        return Vs, a

    def value(self, s):
        """ Compute the optimal value at a state.

            Parameters:
                s   --  The state (index).

            Returns:
                The optimal value at this state.
        """

        Vs, a = self.value_and_action(s)
        return Vs

    def action(self, s):
        """ Compute the optimal action at a belief state.

            Parameters:
                s   --  The state (index).

            Returns:
                The optimal action at this state.
        """

        Vs, a = self.value_and_action(s)
        return a

    def compute_adr(self, mdp, s0, trials=100):
        """ Compute the average discounted reward (ADR) at a given start state.

            Parameters:
                mdp     --  The MDP to compute the ADR.
                s0      --  The initial state index.
                trials  --  The number of trials to average over. Default is 100.

            Returns:
                The ADR value at this state.
        """

        adr = 0.0

        for trial in range(trials):
            s = s0
            discount = 1.0
            discountedReward = 0.0

            for t in range(mdp.horizon):
                a = self.action(s)
                sp = mdp.random_successor(s, a)
                discountedReward += discount * mdp.R[s * mdp.m + a]
                discount *= mdp.gamma
                s = sp

            adr = (float(trial) * adr + discountedReward) / float(trial + 1)

        return adr


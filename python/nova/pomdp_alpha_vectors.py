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
import random

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))
import nova_pomdp_alpha_vectors as npav


class POMDPAlphaVectors(npav.NovaPOMDPAlphaVectors):
    """ The alpha-vector representation of a POMDP policy.

        Specifically, this class is a clean python wrapper around computing the optimal values
        and actions at beliefs, as well as freeing the memory.
    """

    def __init__(self):
        """ The constructor for the POMDPAlphaVectors class. """

        self.n = 0
        self.m = 0
        self.r = 0
        self.Gamma = ct.POINTER(ct.c_float)()
        self.pi = ct.POINTER(ct.c_uint)()

    def __del__(self):
        """ Free the memory of the policy when this object is deleted. """

        result = npav._nova.pomdp_alpha_vectors_uninitialize(self)
        if result != 0:
            print("Failed to free the alpha vectors.")
            raise Exception()

    def __str__(self):
        """ Return the string of the POMDP alpha-vectors.

            Returns:
                The string of the POMDP alpha-vectors.
        """

        result = "r: %i" % (self.r) + "\n\n"

        result += "Gamma:\n%s" % (str(np.array([[self.Gamma[i * self.n + s] \
                    for i in range(self.r)] for s in range(self.n)]))) + "\n\n"

        result += "pi:\n%s" % (str(np.array([self.pi[i] \
                    for i in range(self.r)]))) + "\n\n"

        return result

    def value_and_action(self, b):
        """ Compute the optimal value and action at a belief state.

            Parameters:
                b   --  A numpy array for the belief (n array).

            Returns:
                V   --  The optimal value at this belief.
                a   --  The optimal action at this belief.
        """

        array_type_n_float = ct.c_float * (self.n)
        belief = array_type_n_float(*b)

        Vb = ct.c_float(0.0)
        a = ct.c_uint(0)

        result = npav._nova.pomdp_alpha_vectors_value_and_action(self, belief, ct.byref(Vb), ct.byref(a))
        if result != 0:
            print("Failed to compute the optimal value and action.")
            raise Exception()

        Vb = Vb.value
        a = a.value

        return Vb, a

    def value(self, b):
        """ Compute the optimal value at a belief state.

            Parameters:
                b   --  A numpy array for the belief (n array).

            Returns:
                The optimal value at this belief.
        """

        Vb, a = self.value_and_action(b)
        return Vb

    def action(self, b):
        """ Compute the optimal action at a belief state.

            Parameters:
                b   --  A numpy array for the belief (n array).

            Returns:
                The optimal action at this belief.
        """

        Vb, a = self.value_and_action(b)
        return a

    def compute_adr(self, pomdp, b0, trials=100):
        """ Compute the average discounted reward (ADR) at a given belief.

            Parameters:
                pomdp   --  The POMDP to compute the ADR.
                b0      --  A numpy array for the belief (n array).
                trials  --  The number of trials to average over. Default is 100.

            Returns:
                The ADR value at this belief.
        """

        adr = 0.0

        for trial in range(trials):
            b = b0.copy()
            s = random.choice([i for i in range(pomdp.n) if b0[i] > 0.0])
            discount = 1.0
            discountedReward = 0.0

            for t in range(pomdp.horizon):
                a = self.action(b)
                sp = pomdp.random_successor(s, a)
                o = pomdp.random_observation(a, sp)

                beliefReward = 0.0
                for i in range(pomdp.n):
                    beliefReward += b[i] * pomdp.R[i * pomdp.m + a]
                discountedReward += discount * beliefReward

                discount *= pomdp.gamma
                b = pomdp.belief_update(b, a, o)
                s = sp

            adr = (float(trial) * adr + discountedReward) / float(trial + 1)

        return adr


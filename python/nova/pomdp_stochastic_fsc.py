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
import nova_pomdp_stochastic_fsc as npfsc

import pomdp_alpha_vectors as npav


class POMDPStochasticFSC(npfsc.NovaPOMDPStochasticFSC):
    """ The stochastic FSC representation of a POMDP policy.

        Specifically, this class is a clean python wrapper around using stochastic FSCs,
        as well as freeing the memory once created.

        Note: The initial controller is always assumed to be the first one (index 0).
    """

    def __init__(self):
        """ The constructor for the POMDPStochasticFSC class. """

        self.k = 0
        self.n = 0
        self.m = 0
        self.z = 0
        self.psi = ct.POINTER(ct.c_float)()
        self.eta = ct.POINTER(ct.c_float)()
        self.V = ct.POINTER(ct.c_float)()

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

        result = "k: %i" % (self.k) + "\n"
        result += "n: %i" % (self.n) + "\n"
        result += "m: %i" % (self.m) + "\n"
        result += "z: %i" % (self.z) + "\n\n"

        result += "psi:\n%s" % (str(np.array([[self.psi[x * self.m + a] \
                                            for a in range(self.m)] \
                                        for x in range(self.k)]))) + "\n\n"

        result += "eta:\n%s" % (str(np.array([[[[self.eta[x * self.m * self.z * self.k +
                                                          a * self.z * self.k +
                                                          o * self.k + xp] \
                                            for xp in range(self.k)] \
                                        for o in range(self.z)] \
                                    for a in range(self.m)] \
                                for x in range(self.k)]))) + "\n\n"

        result += "V:\n%s" % (str(np.array([[self.V[x * self.n + s] \
                                            for s in range(self.n)] \
                                        for x in range(self.k)]))) + "\n\n"

        return result

    def save(self, filename):
        """ Save the policy to a file.

            Parameters:
                filename    --  The filename where the policy will be saved.
        """

        with open(filename, 'wb') as f:
            f.write("%i %i %i %i\n" % (self.k, self.n, self.m, self.z))

            for x in range(self.k):
                for a in range(self.m):
                    f.write("%.6f " % (self.psi[x * self.m + a]))
                f.write("\n")

            for x in range(self.k):
                for a in range(self.m):
                    for o in range(self.z):
                        for xp in range(self.k):
                            f.write("%.6f " % (self.eta[x * self.m * self.z * self.k +
                                                        a * self.z * self.k +
                                                        o * self.k + xp]))
                        f.write("\n")

            for x in range(self.k):
                for s in range(self.n):
                    f.write("%.6f " % (self.V[x * self.n + s]))
                f.write("\n")

    def load(self, filename):
        """ Load the policy to a file.

            Parameters:
                filename    --  The filename where the policy will be loaded.
        """

        with open(filename, 'rb') as f:
            result = npfsc._nova.pomdp_stochastic_fsc_uninitialize(self)

            header = f.readline().split()
            self.k = int(header[0])
            self.n = int(header[1])
            self.m = int(header[2])
            self.z = int(header[3])

            result = npfsc._nova.pomdp_stochastic_fsc_initialize(self, self.k, self.n, self.m, self.z)
            if result != 0:
                print("Failed to initialize the stochastic FSC.")
                raise Exception()

            for x in range(self.k):
                line = f.readline().split()
                for a in range(self.m):
                    self.psi[x * self.m + a] = float(line[a])

            for x in range(self.k):
                for a in range(self.m):
                    for o in range(self.z):
                        line = f.readline().split()
                        for xp in range(self.k):
                            self.eta[x * self.m * self.z * self.k +
                                     a * self.z * self.k +
                                     o * self.k + xp] = float(line[xp])

            for x in range(self.k):
                line = f.readline().split()
                for s in range(self.n):
                    self.V[x * self.n + s] = float(line[s])

    def value(self, x, b):
        """ Compute the optimal value at a belief state given a controller node.

            Parameters:
                x   --  The current controller node.
                b   --  A numpy array for the belief (n array).

            Returns:
                The value at this node given the belief.
        """

        # TODO: Implement this.
        return 0.0

        #array_type_n_float = ct.c_float * (self.n)
        #belief = array_type_n_float(*b)

        #Vxb = ct.c_float(0.0)

        #result = npfsc._nova.pomdp_stochastic_fsc_value_and_action(self, x, belief, ct.byref(Vb))
        #if result != 0:
        #    print("Failed to compute the value.")
        #    raise Exception()

        #return Vxb.value

    def random_action(self, x):
        """ Take a random action following psi given the controller node x.

            Parameters:
                x   --  The current controller node.

            Returns:
                A random action selected following psi.
        """

        x = ct.c_uint(x)
        a = ct.c_uint(0)

        result = npfsc._nova.pomdp_stochastic_fsc_random_action(self, x, ct.byref(a))
        if result != 0:
            print("Failed to select a random action.")
            raise Exception()

        return a.value

    def random_successor(self, x, a, o):
        """ Select a random successor following eta given the node, action, and observation.

            Parameters:
                x   --  The current controller node.
                a   --  The action taken at x.
                o   --  The observate made after the action was taken.

            Returns:
                A random successor selected following eta.
        """

        x = ct.c_uint(x)
        a = ct.c_uint(a)
        o = ct.c_uint(o)
        xp = ct.c_uint(0)

        result = npfsc._nova.pomdp_stochastic_fsc_random_successor(self, x, a, o, ct.byref(xp))
        if result != 0:
            print("Failed to select a random successor.")
            raise Exception()

        return xp.value

    def compute_adr(self, pomdp, b0, trials=100):
        """ Compute the average discounted reward (ADR) at a given belief.

            Parameters:
                pomdp               --  The POMDP to compute the ADR.
                b0                  --  A numpy array for the initial belief (n array).
                trials              --  The number of trials to average over. Default is 100.

            Returns:
                The ADR value at this belief.
        """

        adr = 0.0

        for trial in range(trials):
            b = b0.copy()
            s = random.choice([i for i in range(pomdp.n) if b0[i] > 0.0])
            x = 0
            discount = 1.0
            discountedReward = 0.0

            for t in range(pomdp.horizon):
                a = self.random_action(x)
                sp = pomdp.random_successor(s, a)
                o = pomdp.random_observation(a, sp)
                xp = self.random_successor(x, a, o)
                bp = pomdp.belief_update(b, a, o)

                #print("<time, x, a, sp, o, xp, b> = <%i, %i, %i, %i, %i, %i, [%.3f, %.3f]>" % (t, x, a, sp, o, xp, b[0], b[1]))

                # Important: We obtain a reward from the *true* POMDP model,
                # not the FSC model! This is essentially sampling from the
                # true policy tree. So we retain the belief over time
                # and compute the average discounted belief-reward.
                beliefReward = 0.0
                for i in range(pomdp.n):
                    beliefReward += b[i] * pomdp.R[i * pomdp.m + a]
                discountedReward += discount * beliefReward

                discount *= pomdp.gamma
                b = bp
                s = sp
                x = xp

            adr = (float(trial) * adr + discountedReward) / float(trial + 1)

        return adr


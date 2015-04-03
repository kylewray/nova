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

import csv
import numpy as np

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "nova", "wrapper"))
from nova.pomdp import *


class MOPOMDP(object):
    """ A Multi-Objective Partially Observable Markov Decision Process (MOPOMDP) object
        that can load, solve, and save.

        Specifically, it is capable of loading raw and cassandra-like POMDP files, provides
        functionality to solve them using the nova library, and enables saving the resulting
        policy as a raw policy file.
    """

    def __init__(self):
        """ The constructor for the MOPOMDP class. """

        self.n = 0 # The number of states.
        self.m = 0 # The number of actions.
        self.z = 0 # The number of observations.
        self.k = 1 # The number of reward functions.

        self.T = list() # State transitions T(s, a, s').
        self.O = list() # Observation transitions T(a, s', o).
        self.R = list() # Rewards R_i(s, a).

        self.b0 = 0 # The initial belief state.
        self.horizon = 0 # The horizon; non-positive numbers imply infinite horizon.
        self.gamma = 0.9 # The discount factor.

    def load(self, filename):
        """ Load a raw MOPOMDP file given the file.

            Parameters:
                filename    --  The name and path of the file to load.
        """

        # Load all the data in this object.
        data = list()
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                data += [list(row)]

        # Attempt to parse all the data into their respective variables.
        try:
            self.n = int(data[0][0])
            self.m = int(data[0][1])
            self.z = int(data[0][2])
            self.k = int(data[0][3])

            self.s0 = int(data[0][4])
            self.horizon = int(data[0][5])
            self.gamma = float(data[0][6])

            rowOffset = 1
            self.T = np.array([[[float(data[(self.n * a + s) + rowOffset][sp]) \
                                for sp in range(self.n)] \
                            for a in range(self.m)] \
                        for s in range(self.n)])

            rowOffset = 1 + self.n * self.m
            self.O = np.array([[[float(data[(self.n * a + sp) + rowOffset][o]) \
                                for o in range(self.z)] \
                            for sp in range(self.n)] \
                        for a in range(self.m)])


            rowOffset = 1 + self.n * self.m + self.m * self.n
            self.R = np.array([[[float(data[s + rowOffset][a])
                                for a in range(self.m)] \
                            for s in range(self.n)] \
                        for i in range(self.k)])

        except Exception:
            print("Failed to load file.")
            raise Exception()

    def solve(self, f=None, numThreads=1024):
        """ Solve the MOPOMDP, using the nova python wrapper, with the given scalarization function.

            Parameters:
                f           --  The scalarization function which maps a reward vector to a single
                                reward. If set to None, then only the first reward function is
                                used. Default is None. Optional.
                numThreads  --  The number of CUDA threads to execute.

            Returns:
                V   --  The values of each state, mapping states to values.
                pi  --  The policy, mapping states to actions.
        """

        if f == None:
            # The initial value of V is either Rmin / (1 - gamma), for gamma less than 1.0, and
            # simply 0.0 otherwise.
            V = None
            if self.gamma < 1.0:
                Rmin = np.array(self.R).min()
                V = [float(Rmin / (1.0 - self.gamma)) for s in range(self.n)]
            else:
                V = [0.0 for s in range(self.n)]

            pi = [0 for s in range(self.n)]

            numThreads = 1024

            # Call the nova library to run value iteration.
            #result = nova_pomdp_pbvi(self.n, self.m, self.T.flatten(), self.R[0].flatten(),
            #            self.gamma, self.horizon, numThreads, V, pi)
            #if result != 0:
            #    print("Failed to execute nova's solver.")

            return V, pi

        return None

    def __str__(self):
        """ Return the string of the MOPOMDP values akin to the raw file format.

            Returns:
                The string of the MOPOMDP in a similar format as the raw file format.
        """

        result = "n:       " + str(self.n) + "\n"
        result += "m:       " + str(self.m) + "\n"
        result += "z:       " + str(self.z) + "\n"
        result += "k:       " + str(self.k) + "\n"
        result += "b0:      " + str(self.b0) + "\n"
        result += "horizon: " + str(self.horizon) + "\n"
        result += "gamma:   " + str(self.gamma) + "\n\n"

        result += "T(s, a, s'):\n%s" % (str(np.array(self.T))) + "\n\n"
        result += "O(a, s', o):\n%s" % (str(np.array(self.O))) + "\n\n"

        for i in range(self.k):
            result += "R_%i(s, a):\n%s" % (i + 1, str(np.array(self.R[i]))) + "\n\n"

        return result


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

import os
import sys
import pylab

thisFilePath = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.join(thisFilePath, "..", "..", "python"))

from nova.pomdp import *
from nova.pomdp_pbvi import *
from nova.pomdp_perseus import *
from nova.pomdp_hsvi2 import *


files = [
        #{'filename': "tiger_pomdp.raw", 'filetype': "raw", 'process': 'gpu', 'algorithm': 'pbvi', 'expand': None},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'gpu', 'algorithm': 'pbvi', 'expand': "random"},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'gpu', 'algorithm': 'pbvi', 'expand': "random_unique"},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'gpu', 'algorithm': 'pbvi', 'expand': "distinct_beliefs"},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'gpu', 'algorithm': 'pbvi', 'expand': "pema"},

        #{'filename': "tiger_pomdp.raw", 'filetype': "raw", 'process': 'cpu', 'algorithm': 'pbvi', 'expand': None},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'cpu', 'algorithm': 'pbvi', 'expand': "random"},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'cpu', 'algorithm': 'pbvi', 'expand': "random_unique"},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'cpu', 'algorithm': 'pbvi', 'expand': "distinct_beliefs"},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'cpu', 'algorithm': 'pbvi', 'expand': "pema"},

        # NOTE: Not implemented yet.
        #{'filename': "tiger_pomdp.raw", 'filetype': "raw", 'process': 'gpu', 'algorithm': 'perseus', 'expand': None},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'gpu', 'algorithm': 'perseus', 'expand': "random"},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'gpu', 'algorithm': 'perseus', 'expand': "random_unique"},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'gpu', 'algorithm': 'perseus', 'expand': "distinct_beliefs"},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'gpu', 'algorithm': 'perseus', 'expand': "pema"},

        #{'filename': "tiger_pomdp.raw", 'filetype': "raw", 'process': 'cpu', 'algorithm': 'perseus', 'expand': None},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'cpu', 'algorithm': 'perseus', 'expand': "random"},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'cpu', 'algorithm': 'perseus', 'expand': "random_unique"},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'cpu', 'algorithm': 'perseus', 'expand': "distinct_beliefs"},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'cpu', 'algorithm': 'perseus', 'expand': "pema"},

        #{'filename': "tiger_pomdp.raw", 'filetype': "raw", 'process': 'cpu', 'algorithm': 'hsvi2', 'expand': None},
        {'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'cpu', 'algorithm': 'hsvi2', 'expand': None},
        ]

for f in files:
    print("Loading File: %s" % (str(f)))

    tigerFile = os.path.join(thisFilePath, f['filename'])
    tiger = POMDP()
    tiger.load(tigerFile, filetype=f['filetype'])
    #print(tiger)

    if f['expand'] == "random":
        tiger.expand(method=f['expand'], numBeliefsToAdd=250)
    elif f['expand'] == "random_unique":
        tiger.expand(method=f['expand'], numBeliefsToAdd=250, maxTrials=1000)
    elif f['expand'] == "distinct_beliefs":
        for i in range(3):
            tiger.expand(method=f['expand'])
    elif f['expand'] == "pema":
        for i in range(4):
            tiger.expand(method=f['expand'], pemaAlgorithm=POMDPPerseusCPU(tiger))

    print(tiger)

    if f['process'] == "gpu":
        try:
            tiger.initialize_gpu()
        except:
            continue

    if f['algorithm'] == "pbvi" and f['process'] == "cpu":
        algorithm = POMDPPBVICPU(tiger)
    elif f['algorithm'] == "pbvi" and f['process'] == "gpu":
        algorithm = POMDPPBVIGPU(tiger)
    elif f['algorithm'] == "perseus" and f['process'] == "cpu":
        algorithm = POMDPPerseusCPU(tiger)
    elif f['algorithm'] == "hsvi2" and f['process'] == "cpu":
        algorithm = POMDPHSVI2CPU(tiger)
        algorithm.trials = 100
        algorithm.delta = 0.0001
        algorithm.maxAlphaVectors = int(max(tiger.n, tiger.m) + algorithm.trials * tiger.horizon + 1)

    policy = algorithm.solve()
    #print(policy)

    #pylab.hold(True)
    pylab.title("Tiger Alpha-Vectors (Algorithm: %s, Expand: %s)" % (f['algorithm'], f['expand']))
    pylab.xlabel("Belief of State s2: b(s2)")
    pylab.ylabel("Value of Belief: V(b(s2))")
    for i in range(policy.r):
        if policy.pi[i] == 0:
            pylab.plot([0.0, 1.0],
                       [policy.Gamma[i * policy.n + 0], policy.Gamma[i * policy.n + 1]],
                       linewidth=10, color='red')
        elif policy.pi[i] == 1:
            pylab.plot([0.0, 1.0],
                       [policy.Gamma[i * policy.n + 0], policy.Gamma[i * policy.n + 1]],
                       linewidth=10, color='green')
        elif policy.pi[i] == 2:
            pylab.plot([0.0, 1.0],
                       [policy.Gamma[i * policy.n + 0], policy.Gamma[i * policy.n + 1]],
                       linewidth=10, color='blue')
    pylab.show()


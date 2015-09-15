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
from nova.mdp import *
from nova.pomdp import *


files = [
        {'filename': "tiger_pomdp.raw", 'filetype': "raw", 'expand': None},
        {'filename': "tiger_95.pomdp", 'filetype': "pomdp", 'expand': "random"},
        {'filename': "tiger_95.pomdp", 'filetype': "pomdp", 'expand': "distinct_beliefs"},
        {'filename': "tiger_95.pomdp", 'filetype': "pomdp", 'expand': "pema"},
        ]

for f in files:
    tigerFile = os.path.join(thisFilePath, f['filename'])
    tiger = POMDP()
    tiger.load(tigerFile, filetype=f['filetype'])
    print(tiger)

    if f['expand'] == "random":
        # Note: 1 + 250 = 251 belief points.
        tiger.expand(method=f['expand'], numBeliefsToAdd=250)
        print(tiger)
    elif f['expand'] == "distinct_beliefs":
        # Note: 2^3 = 8 belief points.
        for i in range(3):
            tiger.expand(method=f['expand'])
        print(tiger)
    elif f['expand'] == "pema":
        # Note: 1 + 4 = 5 belief points.
        for i in range(4):
            tiger.expand(method=f['expand'])
            print(tiger)

    Gamma, piResult, timing = tiger.solve()
    print(Gamma)
    print(piResult)

    pylab.hold(True)
    pylab.title("Alpha-Vectors for Tiger Problem (Expand: %s)" % (f['expand']))
    pylab.xlabel("Belief of State s2: b(s2)")
    pylab.ylabel("Value of Belief: V(b(s2))")
    for i in range(tiger.r):
        if piResult[i] == 0:
            pylab.plot([0.0, 1.0], [Gamma[i, 0], Gamma[i, 1]], linewidth=10, color='red')
        elif piResult[i] == 1:
            pylab.plot([0.0, 1.0], [Gamma[i, 0], Gamma[i, 1]], linewidth=10, color='green')
        elif piResult[i] == 2:
            pylab.plot([0.0, 1.0], [Gamma[i, 0], Gamma[i, 1]], linewidth=10, color='blue')
    pylab.show()


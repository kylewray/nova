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

thisFilePath = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.join(thisFilePath, "..", "..", "wrapper"))
from nova.mdp import *
from nova.pomdp import *


tigerFile = os.path.join(thisFilePath, "tiger.pomdp")
tiger = MOPOMDP()
tiger.load(tigerFile)
print(tiger)

Gamma, piResult = tiger.solve()
print(Gamma)
print(piResult)


# Note: pylab overwrites 'pi'...
from pylab import *

hold(True)
for i in range(tiger.r):
    if piResult[i] == 0:
        plot([0.0, 1.0], [Gamma[i, 0], Gamma[i, 1]], linewidth=10, color='red')
    elif piResult[i] == 1:
        plot([0.0, 1.0], [Gamma[i, 0], Gamma[i, 1]], linewidth=10, color='green')
    elif piResult[i] == 2:
        plot([0.0, 1.0], [Gamma[i, 0], Gamma[i, 1]], linewidth=10, color='blue')
show()


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

sys.path.append(os.path.join(thisFilePath, "..", "..", "python"))
from nova.pomdp import *

from pylab import *


horizon = 10
numTrials = 1
numBeliefSteps = 1

#processes = ['gpu', 'cpu']
#processes = ['gpu']
processes = ['cpu']

files = [
        {'name': "tiger", 'filename': "domains/tiger_95.pomdp", 'filetype': "pomdp", 'numExpandSteps': 4, 'sigma': False},
        {'name': "tiger", 'filename': "domains/tiger_95.pomdp", 'filetype': "pomdp", 'numExpandSteps': 4, 'sigma': True},
        {'name': "grid-4x3", 'filename': "domains/4x3_95.pomdp", 'filetype': "pomdp", 'numExpandSteps': 5, 'sigma': False},
        {'name': "grid-4x3", 'filename': "domains/4x3_95.pomdp", 'filetype': "pomdp", 'numExpandSteps': 5, 'sigma': True},
        {'name': "tiger-grid", 'filename': "domains/tiger_grid.pomdp", 'filetype': "pomdp", 'numExpandSteps': 7, 'sigma': False},
        {'name': "tiger-grid", 'filename': "domains/tiger_grid.pomdp", 'filetype': "pomdp",'numExpandSteps': 7, 'sigma': True},
        {'name': "hallway2", 'filename': "domains/hallway2.pomdp", 'filetype': "pomdp", 'numExpandSteps': 8, 'sigma': False},
        {'name': "hallway2", 'filename': "domains/hallway2.pomdp", 'filetype': "pomdp", 'numExpandSteps': 8, 'sigma': True},
        {'name': "tag", 'filename': "domains/tag.pomdp", 'filetype': "pomdp", 'numExpandSteps': 8, 'sigma': False},
        {'name': "tag", 'filename': "domains/tag.pomdp", 'filetype': "pomdp", 'numExpandSteps': 8, 'sigma': True},
        {'name': "rock-sample (7x8)", 'filename': "domains/rockSample_7_8.pomdp", 'filetype': "pomdp", 'numExpandSteps': 8, 'sigma': False},
        {'name': "rock-sample (7x8)", 'filename': "domains/rockSample_7_8.pomdp", 'filetype': "pomdp", 'numExpandSteps': 8, 'sigma': True},
        {'name': "auv-navigation", 'filename': "domains/auvNavigation.pomdp", 'filetype': "pomdp", 'numExpandSteps': 8, 'sigma': False},
        {'name': "auv-navigation", 'filename': "domains/auvNavigation.pomdp", 'filetype': "pomdp", 'numExpandSteps': 8, 'sigma': True},
        #{'name': "drive_san_francisco", 'filename': "domains/drive_san_francisco.pomdp", 'filetype': "pomdp", 'numExpandSteps': 8, 'sigma': False},
        #{'name': "drive_san_francisco", 'filename': "domains/drive_san_francisco.pomdp", 'filetype': "pomdp", 'numExpandSteps': 8, 'sigma': True},
        #{'name': "drive_seattle", 'filename': "domains/drive_seattle.pomdp", 'filetype': "pomdp", 'numExpandSteps': 9, 'sigma': False},
        #{'name': "drive_seattle", 'filename': "domains/drive_seattle.pomdp", 'filetype': "pomdp", 'numExpandSteps': 9, 'sigma': True},
        #{'name': "drive_new_york_city", 'filename': "domains/drive_new_york_city.pomdp", 'filetype': "pomdp", 'numExpandSteps': 9, 'sigma': False},
        #{'name': "drive_new_york_city", 'filename': "domains/drive_new_york_city.pomdp", 'filetype': "pomdp", 'numExpandSteps': 9, 'sigma': True},
        #{'name': "drive_boston", 'filename': "domains/drive_boston.pomdp", 'filetype': "pomdp", 'numExpandSteps': 10, 'sigma': False},
        #{'name': "drive_boston", 'filename': "domains/drive_boston.pomdp", 'filetype': "pomdp", 'numExpandSteps': 10, 'sigma': True},
        ]

timings = {f['name']: {'cpu': [0.0 for j in range(numBeliefSteps)],
                       'gpu': [0.0 for j in range(numBeliefSteps)]} for f in files}


for f in files:
    if f['sigma']:
        print(f['name'] + " with sigma-approximation")
    else:
        print(f['name'])

    filename = os.path.join(thisFilePath, f['filename'])

    for p in processes:
        print(" - %s " % (p), end='')

        if f['sigma']:
            fileSuffix = "_".join([f['name'], p, "sigma"])
        else:
            fileSuffix = "_".join([f['name'], p])

        with open(os.path.join(thisFilePath, "results", fileSuffix) + ".csv", "w") as out:
            out.write("n,m,z,r,ns,rz,time,V(b0)\n")

            for j in range(numBeliefSteps):
                #print("Belief Step %i (max %i):" % (f['beliefStepSize'] * (j + 1), (f['beliefStepSize'] * numBeliefSteps)))
                print(".", end='')
                sys.stdout.flush()

                pomdp = POMDP()
                pomdp.load(filename, filetype=f['filetype'])

                # Store the intial belief from this file.
                rzOriginal = pomdp.rz
                b0 = [pomdp.B[0 * pomdp.rz + k] for k in range(pomdp.rz)]
                z0 = [pomdp.Z[0 * pomdp.rz + k] for k in range(pomdp.rz)]

                # Do the expand step a number of times equal to the number of desired belief steps, and optionally
                # do the sigma-approximation of belief.
                for k in range(f['numExpandSteps']):
                    pomdp.expand(method='distinct_beliefs') #, numBeliefsToAdd=(f['beliefStepSize'] * (j + 1)))

                if f['sigma']:
                    #pomdp.sigma_approximate(rz=int(log(pomdp.n) + 1))
                    pomdp.sigma_approximate(rz=int(pomdp.n / 10 + 1))

                pomdp.h = int(horizon)

                Gamma, piResult, timing = pomdp.solve(process=p)

                # Compute the value of the initial belief, stored before doing the sigma approximation.
                Vb0 = 0.0
                for k in range(rzOriginal):
                    s = z0[k]
                    if s < 0:
                        break
                    Vb0 += b0[k] * Gamma[0, s]

                # Note: use the time.time() function, which measures wall-clock time.
                timings[f['name']][p][j] = (float(j) * timings[f['name']][p][j] + timing[0]) / float(j + 1)

                out.write("%i,%i,%i,%i,%i,%i,%.5f,%.5f\n" % (pomdp.n, pomdp.m, pomdp.z, pomdp.r, pomdp.ns, pomdp.rz, timing[0] / float(horizon), Vb0))

        print()



# Plot the results.
#x = [f['beliefStepSize'] * (j + 1) for j in range(numBeliefSteps)]
#
#title("CPU/GPU Execution Time vs. Number of Belief Points")
#xlabel("Number of Belief Points")
#ylabel("Execution Time (seconds)")
#
#xlim([f['beliefStepSize'], numBeliefSteps * f['beliefStepSize']])
#xticks(x)
#
#hold(True)
#
#for f in files:
#    plot(x, timings[f['name']]['cpu'], label=('%s (CPU)' % f['name']))
#    plot(x, timings[f['name']]['gpu'], label=('%s (GPU)' % f['name']))
#
#legend()
#
#show()


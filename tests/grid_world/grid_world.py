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

import itertools as it

thisFilePath = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.join(thisFilePath, "..", "..", "python"))
from nova.mdp import *
from nova.pomdp import *


trials = [
         #{'name': "VI CPU", 'filename': "domains/grid_world_mdp.raw", 'filetype': "raw", 'algorithm': "vi", 'process': "cpu", 'w': 4, 'h': 3},
         #{'name': "VI GPU", 'filename': "domains/grid_world_mdp.raw", 'filetype': "raw", 'algorithm': "vi", 'process': "gpu", 'w': 4, 'h': 3},
         #{'name': "LAO* CPU", 'filename': "domains/grid_world_ssp.raw", 'filetype': "raw", 'algorithm': "lao*", 'process': "cpu", 'w': 4, 'h': 3},

         #{'name': "VI CPU (Intense)", 'filename': "domains/intense_grid_world_mdp.raw", 'filetype': "raw", 'algorithm': "vi", 'process': "cpu", 'w': 50, 'h': 50},
         #{'name': "VI GPU (Intense)", 'filename': "domains/intense_grid_world_mdp.raw", 'filetype': "raw", 'algorithm': "vi", 'process': "gpu", 'w': 50, 'h': 50},
         #{'name': "LAO* CPU (Intense)", 'filename': "domains/intense_grid_world_ssp.raw", 'filetype': "raw", 'algorithm': "lao*", 'process': "cpu", 'w': 50, 'h': 50},

         {'name': "VI CPU (Massive)", 'filename': "domains/massive_grid_world_mdp.raw", 'filetype': "raw", 'algorithm': "vi", 'process': "cpu", 'w': 75, 'h': 75},
         #{'name': "VI GPU (Massive)", 'filename': "domains/massive_grid_world_mdp.raw", 'filetype': "raw", 'algorithm': "vi", 'process': "gpu", 'w': 75, 'h': 75},
         {'name': "LAO* CPU (Massive)", 'filename': "domains/massive_grid_world_ssp.raw", 'filetype': "raw", 'algorithm': "lao*", 'process': "cpu", 'w': 75, 'h': 75},

         #{'name': "PBVI CPU", 'filename': "domains/grid_world_pomdp.raw", 'filetype': "raw", 'algorithm': "pbvi", 'process': "cpu", 'w': 4, 'h': 3},
         #{'name': "PBVI GPU", 'filename': "domains/grid_world_pomdp.raw", 'filetype': "raw", 'algorithm': "pbvi", 'process': "gpu", 'w': 4, 'h': 3},
         ]


for trial in trials:
    print(trial['name'])

    gridWorldFile = os.path.join(thisFilePath, trial['filename'])
    gridWorld = MDP()
    gridWorld.load(gridWorldFile, filetype=trial['filetype'])
    gridWorld.horizon = 10000
    #print(gridWorld)

    # The heuristic (admissible) is the manhattan distance to the goal, which is always the upper right corner.
    #h = np.array([abs(y) + abs(trial['w'] - 1 - x) for y, x in it.product(range(trial['h']), range(trial['w']))] + [0.0]).flatten()
    h = np.array([0.0 for s in range(gridWorld.n)])

    V, pi, timing = gridWorld.solve(algorithm=trial['algorithm'], process=trial['process'], epsilon=0.01, heuristic=h)
    #print([[V[y * trial['w'] + x] for x in range(trial['w'])] for y in range(trial['h'])])
    print([[pi[y * trial['w'] + x] for x in range(trial['w'])] for y in range(trial['h'])])


#gridWorldFile = os.path.join(thisFilePath, "grid_world_pomdp.raw")
#gridWorld = POMDP()
#gridWorld.load(gridWorldFile, filetype='raw')
#print(gridWorld)

#Gamma, pi, timing = gridWorld.solve()
#print(Gamma)
#print(pi)


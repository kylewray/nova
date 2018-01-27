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

import os
import sys
import pylab
import networkx as nx

thisFilePath = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.join(thisFilePath, "..", "..", "python"))

from nova.pomdp import *
from nova.pomdp_pbvi import *
from nova.pomdp_perseus import *
from nova.pomdp_hsvi2 import *
from nova.pomdp_nlp import *
from nova.pomdp_cbnlp import *


files = [
        #{'filename': "tiger_pomdp.raw", 'filetype': "raw", 'process': 'cpu', 'algorithm': 'pbvi', 'expand': None},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'cpu', 'algorithm': 'pbvi', 'expand': "random"},
#        {'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'cpu', 'algorithm': 'pbvi', 'expand': "random_unique"},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'cpu', 'algorithm': 'pbvi', 'expand': "distinct_beliefs"},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'cpu', 'algorithm': 'pbvi', 'expand': "pema"},

        #{'filename': "tiger_pomdp.raw", 'filetype': "raw", 'process': 'gpu', 'algorithm': 'pbvi', 'expand': None},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'gpu', 'algorithm': 'pbvi', 'expand': "random"},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'gpu', 'algorithm': 'pbvi', 'expand': "random_unique"},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'gpu', 'algorithm': 'pbvi', 'expand': "distinct_beliefs"},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'gpu', 'algorithm': 'pbvi', 'expand': "pema"},

        #{'filename': "tiger_pomdp.raw", 'filetype': "raw", 'process': 'cpu', 'algorithm': 'perseus', 'expand': None},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'cpu', 'algorithm': 'perseus', 'expand': "random"},
        {'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'cpu', 'algorithm': 'perseus', 'expand': "random_unique"},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'cpu', 'algorithm': 'perseus', 'expand': "distinct_beliefs"},
        #{'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'cpu', 'algorithm': 'perseus', 'expand': "pema"},

        #{'filename': "tiger_pomdp.raw", 'filetype': "raw", 'process': 'cpu', 'algorithm': 'hsvi2', 'expand': None},
#        {'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'cpu', 'algorithm': 'hsvi2', 'expand': None},

        #{'filename': "tiger_pomdp.raw", 'filetype': "raw", 'process': 'cpu', 'algorithm': 'nlp', 'expand': None},
#        {'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'cpu', 'algorithm': 'nlp', 'expand': "random_unique"},

        #{'filename': "tiger_pomdp.raw", 'filetype': "raw", 'process': 'cpu', 'algorithm': 'cbnlp', 'expand': None},
        {'filename': "tiger_95.pomdp", 'filetype': "cassandra", 'process': 'cpu', 'algorithm': 'cbnlp', 'expand': "random_unique"},
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
            tiger.expand(method=f['expand'], pemaAlgorithm=POMDPPerseus(tiger))

    print(tiger)

    if f['process'] == "gpu":
        try:
            tiger.initialize_gpu()
        except:
            continue

    if f['algorithm'] == "pbvi" and f['process'] == "cpu":
        algorithm = POMDPPBVI(tiger)
    elif f['algorithm'] == "pbvi" and f['process'] == "gpu":
        algorithm = POMDPPBVIGPU(tiger)
    elif f['algorithm'] == "perseus" and f['process'] == "cpu":
        algorithm = POMDPPerseus(tiger)
    elif f['algorithm'] == "hsvi2" and f['process'] == "cpu":
        algorithm = POMDPHSVI2(tiger)
        algorithm.trials = 100
        algorithm.delta = 0.0001
        algorithm.maxAlphaVectors = int(max(tiger.n, tiger.m) + algorithm.trials * tiger.horizon + 1)
    elif f['algorithm'] == "nlp" and f['process'] == "cpu":
        cmd = "python3 " + os.path.join(thisFilePath, "..", "..", "python", "neos_snopt.py") + " "
        cmd += os.path.join(thisFilePath, "nova_nlp_ampl.mod") + " "
        cmd += os.path.join(thisFilePath, "nova_nlp_ampl.dat")
        algorithm = POMDPNLP(tiger, path=thisFilePath, command=cmd, k=5)
    elif f['algorithm'] == "cbnlp" and f['process'] == "cpu":
        cmd = "python3 " + os.path.join(thisFilePath, "..", "..", "python", "neos_snopt.py") + " "
        cmd += os.path.join(thisFilePath, "nova_cbnlp_ampl.mod") + " "
        cmd += os.path.join(thisFilePath, "nova_cbnlp_ampl.dat")
        algorithm = POMDPCBNLP(tiger, path=thisFilePath, command=cmd, k=3, r=5)

    policy = algorithm.solve()
    print(policy)

    if f['algorithm'] in ["pbvi", "perseus", "hsvi2"]:
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
    elif f['algorithm'] in ["nlp", "cbnlp"]:
        G = nx.DiGraph()

        X = ["x%i" % (x) for x in range(policy.k)]
        A = ["x%i-a%i" % (x, a) for x in range(policy.k) for a in range(policy.m)]
        O = ["x%i-a%i-o%i" % (x, a, o) for x in range(policy.k)
                                    for a in range(policy.m) for o in range(policy.z)]

        edges = list()
        psi = list()
        obs = list()
        eta = list()

        psiW = list()
        etaW = list()

        for x in range(policy.k):
            for a in range(policy.m):
                w = policy.psi[x * policy.m + a]
                if w > 0.0 and w <= 1.0:
                    psi += [("x%i" % (x), "x%i-a%i" % (x, a), w)]
                    psiW += [w * 6.0]
                else:
                    A.remove("x%i-a%i" % (x, a))
                    continue

                for o in range(policy.z):
                    obs += [("x%i-a%i" % (x, a), "x%i-a%i-o%i" % (x, a, o))]

                    for xp in range(policy.k):
                        w = policy.eta[x * policy.m * policy.z * policy.k +
                                       a * policy.z * policy.k +
                                       o * policy.k + xp]
                        if w > 0.0 and w <= 1.0:
                            eta += [("x%i-a%i-o%i" % (x, a, o), "x%i" % (xp), w)]
                            etaW += [w * 6.0]

        G.add_weighted_edges_from(psi)
        G.add_edges_from(obs)
        G.add_weighted_edges_from(eta)

        shells = [X, A, O]
        pos = nx.shell_layout(G, shells)
        #pos = nx.circular_layout(G)

        nx.draw_networkx_nodes(G, pos, node_list=X, node_color='k', node_shape='o', node_size=1000)
        nx.draw_networkx_nodes(G, pos, node_list=A, node_color='r', node_shape='s', node_size=1000)
        nx.draw_networkx_nodes(G, pos, node_list=O, node_color='y', node_shape='^', node_size=1000)

        nx.draw_networkx_edges(G, pos, edgelist=psi, width=psiW, alpha=0.75, edge_color='k', arrows=True)
        nx.draw_networkx_edges(G, pos, edgelist=obs, width=2.0, alpha=0.5, edge_color='k', arrows=True)
        nx.draw_networkx_edges(G, pos, edgelist=eta, width=etaW, alpha=0.75, edge_color='k', arrows=True)

        nx.draw_networkx_labels(G, pos, font_size=18, font_family='sans-serif')

        pylab.axis('off')
        pylab.show()


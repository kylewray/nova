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
import csv
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))


class FileLoader(object):
    """ Load a (PO)MDP file in the style of Cassandra's format or a raw format.
    
        Specifically, the variables in the constructor are assigned when loaded.
        These variables are stored as int, float, or numpy arrays, as appropriate.

        Importantly, this only loads rewards with R(s, a), not the more general
        R(s, a, s', o) as in the original format. This is to save on room, and
        since most problem domains really don't need those extra variables.
    """

    def __init__(self):
        """ Initialize the FileLoader class by creating default variables. """

        self.n = 0
        self.ns = 0
        self.m = 0
        self.z = 0
        self.r = 0
        self.rz = 0
        self.k = 0

        self.s0 = 0
        self.ng = 0
        self.goals = None

        self.gamma = 0.0
        self.horizon = 0

        self.Rmin = 0.0
        self.Rmax = 0.0

        self.S = None
        self.T = None
        self.O = None
        self.R = None
        self.Z = None
        self.B = None

        # Note: This is computed to be (Rmax-Rmin)/1000. For example, if rewards are [-100, 10],
        # like in tiger, then epsilon is 0.11.
        self.epsilon = 0.0

    def load_raw_mdp(self, filename, scalarize=lambda x: x[0]):
        """ Load a raw Multi-Objective MDP file given the filename and optionally a scalarization function.

            Parameters:
                filename    --  The name and path of the file to load.
                scalarize   --  Optionally define a scalarization function. Default returns the first reward.
        """

        # Load all the data in this object.
        data = list()
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                data += [list(row)]

        # Attempt to parse all the data into their respective variables.
        try:
            # Load the header information.
            self.n = int(data[0][0])
            self.ns = int(data[0][1])
            self.m = int(data[0][2])

            k = int(data[0][3])
            self.s0 = int(data[0][4])
            self.ng = int(data[0][5])

            self.horizon = int(data[0][6])
            self.gamma = float(data[0][7])

            # Load each of the larger data structures into memory and immediately
            # convert them to their C object type to save memory.
            rowOffset = 1
            self.goals = np.array([int(data[rowOffset][s]) for s in range(self.ng)])

            rowOffset = 2
            self.S = np.array([[[int(data[(self.n * a + s) + rowOffset][sp]) \
                                for sp in range(self.ns)] \
                            for a in range(self.m)] \
                        for s in range(self.n)])

            rowOffset = 2 + self.n * self.m
            self.T = np.array([[[float(data[(self.n * a + s) + rowOffset][sp]) \
                                for sp in range(self.ns)] \
                            for a in range(self.m)] \
                        for s in range(self.n)])

            rowOffset = 2 + self.n * self.m + self.n * self.m
            self.R = scalarize(np.array([[[float(data[(self.m * i + a) + rowOffset][s])
                                for a in range(self.m)] \
                            for s in range(self.n)] \
                        for i in range(k)]))

            self.Rmax = self.R.max()
            self.Rmin = self.R.min()

            # This is computed but not used for computing the horizon in the raw format.
            self.epsilon = (self.Rmax - self.Rmin) / 1000.0

        except Exception:
            print("Failed to load file '%s'." % (filename))
            raise Exception()

    def load_raw_pomdp(self, filename, scalarize=lambda x: x[0]):
        """ Load a raw Multi-Objective POMDP file given the filename and optionally a scalarization function.

            Parameters:
                filename    --  The name and path of the file to load.
                scalarize   --  Optionally define a scalarization function. Default returns the first reward.
        """

        # Load all the data in this object.
        data = list()
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                data += [list(row)]

        # Attempt to parse all the data into their respective variables.
        try:
            # Load the header information.
            self.n = int(data[0][0])
            self.ns = int(data[0][1])
            self.m = int(data[0][2])
            self.z = int(data[0][3])
            self.r = int(data[0][4])
            self.rz = int(data[0][5])
            k = int(data[0][6])

            self.s0 = int(data[0][7])
            self.horizon = int(data[0][8])
            self.gamma = float(data[0][9])

            # Load each of the larger data structures into memory.
            rowOffset = 1
            self.S = np.array([[[int(data[(self.n * a + s) + rowOffset][sp]) \
                                for sp in range(self.ns)] \
                            for a in range(self.m)] \
                        for s in range(self.n)])

            rowOffset = 1 + self.n * self.m
            self.T = np.array([[[float(data[(self.n * a + s) + rowOffset][sp]) \
                                for sp in range(self.ns)] \
                            for a in range(self.m)] \
                        for s in range(self.n)])

            rowOffset = 1 + self.n * self.m + self.n * self.m
            self.O = np.array([[[float(data[(self.z * a + o) + rowOffset][sp]) \
                                for o in range(self.z)] \
                            for sp in range(self.n)] \
                        for a in range(self.m)])

            rowOffset = 1 + self.n * self.m + self.n * self.m + self.m * self.z
            self.R = scalarize(np.array([[[float(data[(self.m * i + a) + rowOffset][s])
                                for a in range(self.m)] \
                            for s in range(self.n)] \
                        for i in range(k)]))

            self.Rmax = self.R.max()
            self.Rmin = self.R.min()

            rowOffset = 1 + self.n * self.m + self.n * self.m + self.m * self.z + k * self.m
            self.Z = np.array([[int(data[i + rowOffset][s])
                            for s in range(self.rz)] \
                        for i in range(self.r)])

            rowOffset = 1 + self.n * self.m + self.n * self.m + self.m * self.z + k * self.m + self.r
            self.B = np.array([[float(data[i + rowOffset][s])
                            for s in range(self.rz)] \
                        for i in range(self.r)])

            # This is computed but not used for computing the horizon in the raw format.
            self.epsilon = (self.Rmax - self.Rmin) / 1000.0

        except Exception:
            print("Failed to load file '%s'." % (filename))
            raise Exception()

    def load_cassandra(self, filename):
        """ Load a Cassandra-format (PO)MDP file given the filename.

            Parameters:
                filename    --  The name and path of the file to load.
        """

        data = self._load_parse(filename)
        pomdp = self._load_extract(filename, data)
        self._load_create(pomdp)

    def _load_parse(self, filename):
        """ Step 1/1: Load the raw data from a file into an easier to use list.

            Parameters:
                filename    --  The name and path of the file to load.

            Returns:
                The list of data points, each a list of relevant information.
        """

        # Load the file into an easier format for parsing later.
        data = list()
        with open(filename, 'r') as f:
            # This variable holds current data for the current variable part being defined.
            # Variables are defined by starting a line with "<keyword>: <data>\n<data>..." etc.
            # until another line starting with "<keyword>:" is found.
            currentDataPoint = list()

            reader = csv.reader(f, delimiter='\n')
            for line in reader:
                # There is no delimiter so each line is either zero or one element.
                if len(line) == 0:
                    continue
                elif len(line) == 1:
                    line = line[0]
                else:
                    print("Failed to load file '%s' due to a bad line: '%s'." % (filename, str(line)))
                    raise Exception()

                # Rip off any '#' comments and clean up white spaces.
                line = line.split('#')[0].strip()
                if len(line) == 0:
                    continue

                # If it is a variable definition statement, then we have just ended
                # the previous variable definition, so add it to the data variable
                # and prepare to start again.
                if ":" in line:
                    data += [currentDataPoint]
                    currentDataPoint = list()

                currentDataPoint += [line]

            data += [currentDataPoint]

        # There is an extra one at the start.
        return data[1:]

    def _load_extract(self, filename, data):
        """ Step 2/3: Given the parsed raw data, create a mapping from variables to data.

            Parameters:
                filename    --  The name and path of the file to load.
                data        --  A list of data points, each a list of related variable information.

            Returns:
                A dictionary containing variable-data mappings for the POMDP.
        """

        # Attempt to parse all the data into their respective variables.
        pomdp = {'values': "reward"}
        try:
            for d in data:
                # The first element always has the variable being defined.
                variable = d[0].split(":")[0].strip()

                # The remaining elements help specify what the rest of the data is.
                parameters = [x.strip() for x in d[0].split(":")[1:] if len(x) > 0]

                # If we have no parameters, then all the relevant data is stored on the next elements in d.
                # Split each of these elements by spaces, and trim white space. This is our data! We are done!
                if len(parameters) == 0:
                    if len(d) == 2:
                        pomdp[variable] = [x.strip() for x in d[1].split() if len(x.strip()) > 0]
                    else:
                        pomdp[variable] = [[x.strip() for x in di.split() if len(x.strip()) > 0] \
                                                                for di in d[1:]]
                    continue

                # If we have parameters, we will now handle the case when we have valid data at the end,
                # versus having the data afterwards in d[1:]. More concretely, if d has only one element, then
                # data is stored on this line, which needs to be assigned. Otherwise, data is stored on the
                # proceeding elements in d[1:], but we have an extra tuple of tokens which specify this data.

                # Take the last parameter and check if it is a single token or a list of tokens.
                tokenOrList = [x.strip() for x in parameters[-1].split() if len(x.strip()) > 0]

                if len(d) == 1 and len(parameters) == 1:
                    pomdp[variable] = tokenOrList

                if len(d) == 1 and len(parameters) > 1:
                    try:
                        pomdp[variable][tuple([x for x in parameters[0:-1]] + [tokenOrList[0]])] = \
                                tokenOrList[1]
                    except KeyError:
                        pomdp[variable] = dict()
                        pomdp[variable][tuple([x for x in parameters[0:-1]] + [tokenOrList[0]])] = \
                                tokenOrList[1]

                if len(d) == 2:
                    try:
                        pomdp[variable][tuple(parameters)] = \
                                [x.strip() for x in d[1].split() if len(x.strip()) > 0]
                    except KeyError:
                        pomdp[variable] = dict()
                        pomdp[variable][tuple(parameters)] = \
                                [x.strip() for x in d[1].split() if len(x.strip()) > 0]

                if len(d) > 2:
                    try:
                        pomdp[variable][tuple(parameters)] = \
                                [[x.strip() for x in di.split() if len(x.strip()) > 0] for di in d[1:]]
                    except KeyError:
                        pomdp[variable] = dict()
                        pomdp[variable][tuple(parameters)] = \
                                [[x.strip() for x in di.split() if len(x.strip()) > 0] for di in d[1:]]

        except Exception:
            print("Failed to load file '%s'." % (filename))
            raise Exception()

        return pomdp

    def _load_create(self, pomdp):
        """ Step 3/3: Given the pomdp variable-parameter mapping, load and create the POMDP.

            Parameters:
                pomdp   --  The dictionary mapping keyword variables to parameter data.
        """

        # For each variable (e.g., "states", "T", etc.), create it as the final C object.
        try:
            self.gamma = float(pomdp['discount'][0])
        except KeyError:
            self.gamma = 0.9

        try:
            if len(pomdp['states']) == 1:
                self.n = int(pomdp['states'][0])
                pomdp['states'] = [str(i) for i in range(self.n)]
            else:
                self.n = len(pomdp['states'])
        except KeyError:
            pass

        try:
            if len(pomdp['actions']) == 1:
                self.m = int(pomdp['actions'][0])
                pomdp['actions'] = [str(i) for i in range(self.m)]
            else:
                self.m = len(pomdp['actions'])
        except KeyError:
            pass

        try:
            if len(pomdp['observations']) == 1:
                self.z = int(pomdp['observations'][0])
                pomdp['observations'] = [str(i) for i in range(self.z)]
            else:
                self.z = len(pomdp['observations'])
        except KeyError:
            pass

        try:
            if len(pomdp['start']) == 1 and pomdp['start'] == "uniform":
                self.r = 1
                self.rz = self.n

                self.Z = np.array([[int(s)
                                    for s in range(self.rz)] \
                                for i in range(self.r)])
                self.B = np.array([[1.0 / float(self.rz)
                                    for s in range(self.rz)] \
                                for i in range(self.r)])
            elif len(pomdp['start']) == 1:
                self.r = 1
                self.rz = 1

                self.Z = np.array([[-1
                                    for s in range(self.rz)] \
                                for i in range(self.r)])
                self.B = np.array([[0.0
                                    for s in range(self.rz)] \
                                for i in range(self.r)])
                self.Z[0] = int(pomdp['states'].index(pomdp['start'][0]))
                self.B[0] = float(1.0)
            else:
                self.r = 1
                self.rz = self.n

                self.Z = np.array([[int(s)
                                    for s in range(self.rz)] \
                                for i in range(self.r)])
                self.B = np.array([[float(pomdp['start'][s])
                                    for s in range(self.rz)] \
                                for i in range(self.r)])
        except KeyError:
            pass

        try:
            self.r = 1
            self.rz = len(pomdp['start include'])

            self.Z = np.array([[int(pomdp['states'].index(pomdp['start include'][s]))
                            for s in range(self.rz)] \
                        for i in range(self.r)])
            self.B = np.array([[1.0 / float(self.rz)
                                for s in range(self.rz)] \
                            for i in range(self.r)])
        except KeyError:
            pass

        try:
            self.r = 1
            self.rz = self.n - len(pomdp['start exclude'])

            self.Z = np.array([[int(s)
                            for s in range(self.n) if pomdp['states'][s] not in pomdp['start exclude']] \
                        for i in range(self.r)])
            self.B = np.array([[1.0 / float(self.rz)
                                for s in range(self.rz)] \
                            for i in range(self.r)])
        except KeyError:
            pass

        try:
            S = [[list() for a in range(self.m)] for s in range(self.n)]
            T = [[list() for a in range(self.m)] for s in range(self.n)]

            for key, value in pomdp['T'].items():
                actions = list(range(self.m))
                if key[0] != '*':
                    actions = [pomdp['actions'].index(key[0])]

                if len(key) == 1:
                    if value[0] == "uniform":
                        value = [[1.0 / self.n for sp in range(self.n)] for s in range(self.n)]
                    elif value[0] == "identity":
                        value = [[float(s == sp) for sp in range(self.n)] for s in range(self.n)]

                    for a in actions:
                        for s in range(self.n):
                            for sp in range(self.n):
                                if float(value[s][sp]) > 0.0:
                                    S[s][a] += [int(sp)]
                                    T[s][a] += [float(value[s][sp])]
                elif len(key) == 2:
                    if value[0] == "uniform":
                        value = [1.0 / self.n for sp in range(self.n)]

                    states = list(range(self.n))
                    if key[1] != '*':
                        states = [pomdp['states'].index(key[1])]

                    for a in actions:
                        for s in states:
                            for sp in range(self.n):
                                if float(value[sp]) > 0.0:
                                    S[s][a] += [int(sp)]
                                    T[s][a] += [float(value[sp])]
                elif len(key) == 3:
                    states = list(range(self.n))
                    if key[1] != '*':
                        states = [pomdp['states'].index(key[1])]

                    statePrimes = list(range(self.n))
                    if key[2] != '*':
                        statePrimes = [pomdp['states'].index(key[2])]

                    for a in actions:
                        for s in states:
                            for sp in statePrimes:
                                if float(value) > 0.0:
                                    S[s][a] += [int(sp)]
                                    T[s][a] += [float(value)]

            self.ns = 1
            for s in range(self.n):
                for a in range(self.m):
                    self.ns = max(self.ns, len(T[s][a]))

            # Fill in the remaining elements with -1 and 0.0 accordingly.
            for s in range(self.n):
                for a in range(self.m):
                    for i in range(self.ns - len(S[s][a])):
                        S[s][a] += [int(-1)]
                        T[s][a] += [float(0.0)]

            self.S = np.array(S)
            self.T = np.array(T)
        except KeyError:
            pass

        try:
            self.O = np.zeros((self.m, self.n, self.z))

            for key, value in pomdp['O'].items():
                actions = list(range(self.m))
                if key[0] != '*':
                    actions = [pomdp['actions'].index(key[0])]

                if len(key) == 1:
                    if value[0] == "uniform":
                        value = [[1.0 / self.z for o in range(self.z)] for sp in range(self.n)]

                    for a in actions:
                        for sp in range(self.n):
                            for o in range(self.z):
                                self.O[a, sp, o] = float(value[sp][o])
                elif len(key) == 2:
                    if value[0] == "uniform":
                        value = [1.0 / self.z for o in range(self.z)]

                    statePrimes = list(range(self.n))
                    if key[1] != '*':
                        statePrimes = [pomdp['states'].index(key[1])]

                    for a in actions:
                        for sp in statePrimes:
                            for o in range(self.z):
                                self.O[a, sp, o] = float(value[o])
                elif len(key) == 3:
                    statePrimes = list(range(self.n))
                    if key[1] != '*':
                        statePrimes = [pomdp['states'].index(key[1])]

                    observations = list(range(self.z))
                    if key[2] != '*':
                        observations = [pomdp['observations'].index(key[2])]

                    for a in actions:
                        for sp in statePrimes:
                            for o in observations:
                                self.O[a, sp, o] = float(value)

        except KeyError:
            pass

        try:
            self.k = 1

            self.R = np.zeros((self.n, self.m))
            for key, value in pomdp['R'].items():
                actions = list(range(self.m))
                if key[0] != '*':
                    actions = [pomdp['actions'].index(key[0])]

                states = list(range(self.n))
                if key[1] != '*':
                    states = [pomdp['states'].index(key[1])]

                # Note that 'value' can be a matrix, vector, or value, but we take the
                # average and that 'value' may be defined else where and this could override
                # information.
                for a in actions:
                    for s in states:
                        self.R[s, a] = np.array([value]).astype(np.float).mean()

            # Store the minimum R for use in algorithms.
            self.Rmax = self.R.max()
            self.Rmin = self.R.min()

            # Compute the optimal horizon so that we are within epsilon of the optimal values V.
            self.epsilon = (self.Rmax - self.Rmin) / 1000.0
            self.horizon = int(np.log(self.epsilon / (self.Rmax - self.Rmin)) / np.log(self.gamma)) + 1

        except KeyError:
            pass


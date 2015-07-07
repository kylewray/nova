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

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))
import nova_pomdp as npm


class POMDP(npm.NovaPOMDP):
    """ A Partially Observable Markov Decision Process (POMDP) object that can load, solve, and save.

        Specifically, it is capable of loading raw and cassandra-like POMDP files, provides
        functionality to solve them using the nova library, and enables saving the resulting
        policy as a raw policy file.
    """

    def __init__(self):
        """ The constructor for the POMDP class. """

        # Assign a nullptr for the device-side pointers. These will be set if the GPU is utilized.
        self.currentHorizon = int(0)
        self.Gamma = ct.POINTER(ct.c_float)()
        self.GammaPrime = ct.POINTER(ct.c_float)()
        self.pi = ct.POINTER(ct.c_uint)()
        self.d_S = ct.POINTER(ct.c_int)()
        self.d_T = ct.POINTER(ct.c_float)()
        self.d_O = ct.POINTER(ct.c_float)()
        self.d_R = ct.POINTER(ct.c_float)()
        self.d_Z = ct.POINTER(ct.c_int)()
        self.d_B = ct.POINTER(ct.c_float)()
        self.d_Gamma = ct.POINTER(ct.c_float)()
        self.d_GammaPrime = ct.POINTER(ct.c_float)()
        self.d_pi = ct.POINTER(ct.c_uint)()
        self.d_alphaBA = ct.POINTER(ct.c_float)()

        # Additional informative variables.
        self.Rmin = None
        self.Rmax = None
        self.epsilon = 0.01

    def load(self, filename, filetype='pomdp', scalarize=lambda x: x[0]):
        """ Load a raw Multi-Objective POMDP file given the filename and optionally the file type.

            Parameters:
                filename    --  The name and path of the file to load.
                filetype    --  Either 'pomdp' or 'raw'. Default is 'pomdp'.
                scalarize   --  Optionally define a scalarization function. Only used for 'raw' files.
                                Default returns the first reward.
        """

        if filetype == 'pomdp':
            self._load_pomdp(filename)
        elif filetype == 'raw':
            self._load_raw(filename, scalarize)
        else:
            print("Invalid file type '%s'." % (filetype))
            raise Exception()

    def _load_pomdp(self, filename):
        """ Load a Cassandra-format POMDP file given the filename and optionally a scalarization function.

            Parameters:
                filename    --  The name and path of the file to load.
        """

        data = self._load_pomdp_parse(filename)
        pomdp = self._load_pomdp_extract(filename, data)
        self._load_pomdp_create(pomdp)

    def _load_pomdp_parse(self, filename):
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

    def _load_pomdp_extract(self, filename, data):
        """ Step 2/3: Given the parsed raw data, create a mapping from variables to data.

            Parameters:
                filename    --  The name and path of the file to load.
                data    --  A list of data points, each a list of related variable information.

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

    def _load_pomdp_create(self, pomdp):
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

                array_type_rrz_int = ct.c_int * (self.r * self.rz)
                array_type_rrz_float = ct.c_float * (self.r * self.rz)

                self.Z = array_type_rrz_int(*np.array([[int(s)
                                    for s in range(self.rz)] \
                                for i in range(self.r)]).flatten())
                self.B = array_type_rrz_float(*np.array([[1.0 / float(self.rz)
                                    for s in range(self.rz)] \
                                for i in range(self.r)]).flatten())
            elif len(pomdp['start']) == 1:
                self.r = 1
                self.rz = 1

                array_type_rrz_int = ct.c_int * (self.r * self.rz)
                array_type_rrz_float = ct.c_float * (self.r * self.rz)

                self.Z = array_type_rrz_int(*np.array([[int(pomdp['states'].index(pomdp['start'][0]))
                                    for s in range(self.rz)] \
                                for i in range(self.r)]).flatten())
                self.B = array_type_rrz_float(*np.array([[1.0
                                    for s in range(self.rz)] \
                                for i in range(self.r)]).flatten())
            else:
                self.r = 1
                self.rz = self.n

                array_type_rrz_int = ct.c_int * (self.r * self.rz)
                array_type_rrz_float = ct.c_float * (self.r * self.rz)

                self.Z = array_type_rrz_int(*np.array([[int(s)
                                    for s in range(self.rz)] \
                                for i in range(self.r)]).flatten())
                self.B = array_type_rrz_float(*np.array([[float(pomdp['start'][s])
                                    for s in range(self.rz)] \
                                for i in range(self.r)]).flatten())
        except KeyError:
            pass

        try:
            self.r = 1
            self.rz = len(pomdp['start include'])

            array_type_rrz_int = ct.c_int * (self.r * self.rz)
            array_type_rrz_float = ct.c_float * (self.r * self.rz)

            self.Z = array_type_rrz_int(*np.array([[int(pomdp['states'].index(pomdp['start include'][s]))
                            for s in range(self.rz)] \
                        for i in range(self.r)]).flatten())
            self.B = array_type_rrz_float(*np.array([[1.0 / float(self.rz)
                                for s in range(self.rz)] \
                            for i in range(self.r)]).flatten())
        except KeyError:
            pass

        try:
            self.r = 1
            self.rz = self.n - len(pomdp['start exclude'])

            array_type_rrz_int = ct.c_int * (self.r * self.rz)
            array_type_rrz_float = ct.c_float * (self.r * self.rz)

            self.Z = array_type_rrz_int(*np.array([[int(s)
                            for s in range(self.n) if pomdp['states'][s] not in pomdp['start exclude']] \
                        for i in range(self.r)]).flatten())
            self.B = array_type_rrz_float(*np.array([[1.0 / float(self.rz)
                                for s in range(self.rz)] \
                            for i in range(self.r)]).flatten())
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
                        T[s][a] += [int(0.0)]

            array_type_nmns_int = ct.c_int * (self.n * self.m * self.ns)
            array_type_nmns_float = ct.c_float * (self.n * self.m * self.ns)

            self.S = array_type_nmns_int(*np.array(S).flatten())
            self.T = array_type_nmns_float(*np.array(T).flatten())
        except KeyError:
            pass

        try:
            array_type_mnz_float = ct.c_float * (self.m * self.n * self.z)

            O = np.zeros((self.m, self.n, self.z))
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
                                O[a, sp, o] = float(value[sp][o])
                elif len(key) == 2:
                    if value[0] == "uniform":
                        value = [1.0 / self.z for o in range(self.z)]

                    statePrimes = list(range(self.n))
                    if key[1] != '*':
                        statePrimes = [pomdp['states'].index(key[1])]

                    for a in actions:
                        for sp in statePrimes:
                            for o in range(self.z):
                                O[a, sp, o] = float(value[o])
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
                                O[a, sp, o] = float(value)

            self.O = array_type_mnz_float(*O.flatten())
        except KeyError:
            pass

        try:
            self.k = 1

            array_type_nm_float = ct.c_float * (self.n * self.m)

            R = np.zeros((self.n, self.m))
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
                        R[s, a] = np.array([value]).astype(np.float).mean()

            # Store the minimum R for use in algorithms.
            self.Rmax = R.max()
            self.Rmin = R.min()

            # Compute the optimal horizon so that we are within epsilon of the optimal values V.
            self.horizon = int(np.log(self.epsilon / (self.Rmax - self.Rmin)) / np.log(self.gamma)) + 1

            self.R = array_type_nm_float(*R.flatten())

        except KeyError:
            pass

    def _load_raw(self, filename, scalarize=lambda x: x[0]):
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
            self.k = int(data[0][6])

            self.horizon = int(data[0][7])
            self.gamma = float(data[0][8])

            # Functions to convert flattened NumPy arrays to C arrays.
            array_type_nmns_int = ct.c_int * (self.n * self.m * self.ns)
            array_type_nmns_float = ct.c_float * (self.n * self.m * self.ns)
            array_type_mnz_float = ct.c_float * (self.m * self.n * self.z)
            array_type_nm_float = ct.c_float * (self.n * self.m)
            array_type_rrz_int = ct.c_int * (self.r * self.rz)
            array_type_rrz_float = ct.c_float * (self.r * self.rz)

            # Load each of the larger data structures into memory and immediately
            # convert them to their C object type to save memory.
            rowOffset = 1
            self.S = array_type_nmns_int(*np.array([[[int(data[(self.n * a + s) + rowOffset][sp]) \
                                for sp in range(self.ns)] \
                            for a in range(self.m)] \
                        for s in range(self.n)]).flatten())

            rowOffset = 1 + self.n * self.m
            self.T = array_type_nmns_float(*np.array([[[float(data[(self.n * a + s) + rowOffset][sp]) \
                                for sp in range(self.ns)] \
                            for a in range(self.m)] \
                        for s in range(self.n)]).flatten())

            rowOffset = 1 + self.n * self.m + self.n * self.m
            self.O = array_type_mnz_float(*np.array([[[float(data[(self.z * a + o) + rowOffset][sp]) \
                                for o in range(self.z)] \
                            for sp in range(self.n)] \
                        for a in range(self.m)]).flatten())

            rowOffset = 1 + self.n * self.m + self.n * self.m + self.m * self.z
            self.R = array_type_nm_float(*scalarize(np.array([[[float(data[(self.m * i + a) + rowOffset][s])
                                for a in range(self.m)] \
                            for s in range(self.n)] \
                        for i in range(self.k)])).flatten())

            self.Rmax = max([self.R[i] for i in range(self.n * self.m)])
            self.Rmin = min([self.R[i] for i in range(self.n * self.m)])

            rowOffset = 1 + self.n * self.m + self.n * self.m + self.m * self.z + self.k * self.m
            self.Z = array_type_rrz_int(*np.array([[int(data[i + rowOffset][s])
                            for s in range(self.rz)] \
                        for i in range(self.r)]).flatten())

            rowOffset = 1 + self.n * self.m + self.n * self.m + self.m * self.z + self.k * self.m + self.r
            self.B = array_type_rrz_float(*np.array([[float(data[i + rowOffset][s])
                            for s in range(self.rz)] \
                        for i in range(self.r)]).flatten())

        except Exception:
            print("Failed to load file '%s'." % (filename))
            raise Exception()

    def expand(self, method='random', numDesiredBeliefPoints=1000):
        """ Expand the belief points by either PBVI methods or Perseus' random method.

            Parameters:
                method                  --  The method to use for expanding belief points. Default is 'random'.
                                            Methods:
                                                'random'    Random trajectories through the belief space.
                                                'ger'       A Greedy Error Reduction (GER) exploration.
                numDesiredBeliefPoints  --  Optionally define the number of belief points. Used by the
                                            'random' method. Default is 1000.
        """

        array_type_uint = ct.c_uint * (1)
        array_type_rrz_float = ct.c_float * (numDesiredBeliefPoints * self.n)

        maxNonZeroValues = array_type_uint(*np.array([0]))
        Bnew = array_type_rrz_float(*np.zeros(numDesiredBeliefPoints * self.n))

        if method == "random":
            npm._nova.pomdp_pbvi_expand_random_cpu(self, numDesiredBeliefPoints, maxNonZeroValues, Bnew)
        #elif method == "ger":
        #    npm._nova.pomdp_pbvi_expand_ger_cpu(self)

        # Reconstruct the compressed Z and B.
        self.r = int(numDesiredBeliefPoints)
        self.rz = int(maxNonZeroValues[0])

        array_type_rrz_int = ct.c_int * (self.r * self.rz)
        array_type_rrz_float = ct.c_float * (self.r * self.rz)

        self.Z = array_type_rrz_int(*np.zeros(self.r * self.rz).astype(int))
        self.B = array_type_rrz_float(*np.zeros(self.r * self.rz).astype(float))

        for i in range(self.r):
            j = 0
            for s in range(self.n):
                if Bnew[i * self.n + s] > 0.0:
                    self.Z[i * self.rz + j] = s
                    self.B[i * self.rz + j] = Bnew[i * self.n + s]
                    j += 1

    def solve(self, numThreads=1024, method='pbvi', epsilon=None):
        """ Solve the POMDP using the nova Python wrapper.

            Parameters:
                numThreads  --  The number of CUDA threads to execute (multiple of 32). Default is 1024.
                method      --  The method to use, either 'pbvi' or 'perseus'. Default is 'pbvi'.
                epsilon     --  The error of the value function, changing the horizon. Default is 'None'.

            Returns:
                Gamma   --  The alpha-vectors, one for each belief point, mapping states to values.
                pi      --  The policy, mapping alpha-vectors (belief points) to actions.
        """

        # If epsilon is specified, then re-assign the horizon.
        if epsilon is not None and epsilon > 0.0:
            self.horizon = np.log(epsilon / (self.Rmax - self.Rmin)) / np.log(self.gamma)
            
        # Create Gamma and pi, assigning them their respective initial values.
        Gamma = np.array([[0.0 for s in range(self.n)] for b in range(self.r)])
        if self.gamma < 1.0:
            Gamma = np.array([[float(self.Rmin / (1.0 - self.gamma)) for s in range(self.n)] \
                        for i in range(self.r)])
        Gamma = Gamma.flatten()
        pi = np.array([0 for i in range(self.r)])

        # Create functions to convert flattened NumPy arrays to C arrays.
        array_type_rn_float = ct.c_float * (self.r * self.n)
        array_type_r_uint = ct.c_uint * self.r

        # Create C arrays for the result.
        GammaResult = array_type_rn_float(*Gamma)
        piResult = array_type_r_uint(*pi)

        # Solve the POMDP!
        result = npm._nova.pomdp_pbvi_complete_gpu(self, int(numThreads), GammaResult, piResult)
        if result != 0:
            result = npm._nova.pomdp_pbvi_complete_cpu(self, GammaResult, piResult)

        if result == 0:
            Gamma = np.array([GammaResult[i] for i in range(self.r * self.n)])
            pi = np.array([piResult[i] for i in range(self.r)])
        else:
            print("Failed to solve POMDP using the 'nova' library.")
            raise Exception()

        Gamma = Gamma.reshape((self.r, self.n))

        return Gamma, pi

    def __str__(self):
        """ Return the string of the MOPOMDP values akin to the raw file format.

            Returns:
                The string of the MOPOMDP in a similar format as the raw file format.
        """

        result = "n:       " + str(self.n) + "\n"
        result += "ns:      " + str(self.ns) + "\n"
        result += "m:       " + str(self.m) + "\n"
        result += "z:       " + str(self.z) + "\n"
        result += "r:       " + str(self.r) + "\n"
        result += "rz:      " + str(self.rz) + "\n"
        result += "k:       " + str(self.k) + "\n"
        result += "horizon: " + str(self.horizon) + "\n"
        result += "gamma:   " + str(self.gamma) + "\n"
        result += "epsilon: " + str(self.epsilon) + "\n\n"

        result += "S(s, a, s'):\n%s" % (str(np.array([self.S[i] \
                    for i in range(self.n * self.m * self.ns)]).reshape((self.n, self.m, self.ns)))) + "\n\n"

        result += "T(s, a, s'):\n%s" % (str(np.array([self.T[i] \
                    for i in range(self.n * self.m * self.ns)]).reshape((self.n, self.m, self.ns)))) + "\n\n"

        result += "O(a, s', o):\n%s" % (str(np.array([self.O[i] \
                    for i in range(self.m * self.n * self.z)]).reshape((self.m, self.n, self.z)))) + "\n\n"

        result += "R(s, a):\n%s" % (str(np.array([self.R[i] \
                    for i in range(self.n * self.m)]).reshape((self.n, self.m)))) + "\n\n"

        result += "Z(i, s):\n%s" % (str(np.array([self.Z[i] \
                    for i in range(self.r * self.rz)]).reshape((self.r, self.rz)))) + "\n\n"

        result += "B(i, s):\n%s" % (str(np.array([self.B[i] \
                    for i in range(self.r * self.rz)]).reshape((self.r, self.rz)))) + "\n\n"

        return result


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


import sys
import time

import xmlrpc.server
import xmlrpc.client


class NEOSSNOPT(object):
    """ Connect to the NEOS server, send the model/data files, run SNOPT, and print. """

    def __init__(self, modelFile, dataFile):
        """ The constructor for the NEOS class.

            Parameters:
                modelFile       --  The model file path.
                dataFile        --  The data file path.
        """

        self.modelFile = modelFile
        self.dataFile = dataFile

        self.result = ""

    def _connect(self):
        """ Connect to NEOS and verify that things are working. """

        NEOS_HOST="neos-server.org"
        NEOS_PORT=3333

        self.neos = xmlrpc.client.ServerProxy("https://%s:%d" % (NEOS_HOST, NEOS_PORT))

        if self.neos.ping() != "NeosServer is alive\n":
            print("NEOS SNOPT Error: Failed to make a connection to the NEOS server...")
            raise Exception()

    def _submit(self):
        """ Submit a job to NEOS server to solves the program. """

        self.result = ""

        # Load the model, data, and commands files.
        model = open(self.modelFile).read()
        data = open(self.dataFile).read()
        commands = "solve; option snopt_options 'timing=1 wantsol=2'; display _varname, _var;"

        # Construct the XML string for the job submission.
        xmlString = "<document>\n"
        xmlString += "<category>nco</category><solver>SNOPT</solver><inputMethod>AMPL</inputMethod>\n"
        xmlString += "<model><![CDATA[%s]]></model>\n" % (model)
        xmlString += "<data><![CDATA[%s]]></data>\n" % (data)
        xmlString += "<commands><![CDATA[%s]]></commands>\n" % (commands)
        xmlString += "</document>"

        # Submit the job. If an error happens, handle it. Otherwise, wait for the job to complete.
        (jobNumber, jobPassword) = self.neos.submitJob(xmlString)

        if jobNumber == 0:
            print("NEOS SNOPT Error: Failed to submit job, probably because there have too many.")
            raise Exception()

        sys.stdout.flush()

        # Continuously check if the job is done. Note: The getIntermediateResults function will
        # intentionally hang until a new packet is received from NEOS server.
        offset = 0
        status = ""

        while status != "Done":
            status = self.neos.getJobStatus(jobNumber, jobPassword)
            time.sleep(1)
            #print('.', end='')
            #sys.stdout.flush()

        # Note: We need to give the server time to write the output of the solver for us to read.
        time.sleep(3)
        msg = self.neos.getFinalResults(jobNumber, jobPassword)
        self.result = msg.data.decode()

        time.sleep(1)
        #print("Done!")

    def solve(self):
        """ Call the NEOS server and run SNOPT. Cleanup the output afterwards for nova.

            Returns:
                The cleaned up result of the solver.
        """

        result = ""

        neosSnopt._connect()
        neosSnopt._submit()

        output = self.result.split(" iterations, objective ")[1].split("\n")

        counter = 3
        line = ""

        while counter < len(output):
            line = list(filter(lambda x: x != "", output[counter].split(" ")))
            if line[0] == ";":
                break

            param = line[1][1:-1]

            if param[0:6] == "policy":
                var = [int(v) - 1 for v in param[7:-1].split(',')]
                value = max(0.0, min(1.0, float(line[2])))

                if len(var) == 4: # and value > 0.0:
                    result += "%i %i %i %i %.5f\n" % (var[0], var[1], var[2], var[3], value)

            counter += 1

        return result


if __name__ == "__main__":
    if len(sys.argv) == 3:
        neosSnopt = NEOSSNOPT(sys.argv[1], sys.argv[2])
        print(neosSnopt.solve())
    else:
        print("NEOS SNOPT Error: Format: python3 neos.py <model file path> <data file path>")
        sys.exit(1)


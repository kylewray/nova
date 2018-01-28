/**
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2017 Kyle Hollins Wray, University of Massachusetts
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy of
 *  this software and associated documentation files (the "Software"), to deal in
 *  the Software without restriction, including without limitation the rights to
 *  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *  the Software, and to permit persons to whom the Software is furnished to do so,
 *  subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *  FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 *  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 *  IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 *  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */


#include <nova/pomdp/utilities/pomdp_ampl.h>

#include <nova/error_codes.h>
#include <nova/constants.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <sstream>

namespace nova {

int pomdp_ampl_save_data_file(const POMDP *pomdp, unsigned int k, unsigned int r,
        const char *path, const char *filename)
{
    if (path == nullptr) {
        fprintf(stderr, "Error[pomdp_ampl_save_data_file]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    std::string pathAndFilename(path);
    pathAndFilename += "/";
    pathAndFilename += filename;

    std::ofstream file(pathAndFilename, std::ofstream::out);
    if (!file.is_open()) {
        return NOVA_ERROR_FAILED_TO_OPEN_FILE;
    }

    file << "data;" << std::endl;
    file << std::endl;

    file << "let NUM_STATES := " << pomdp->n << ";" << std::endl;
    file << "let NUM_ACTIONS := " << pomdp->m << ";" << std::endl;
    file << "let NUM_OBSERVATIONS := " << pomdp->z << ";" << std::endl;
    file << "let NUM_NODES := " << k << ";" << std::endl;
    if (r > 1) {
        file << "let NUM_BELIEFS := " << pomdp->r << ";" << std::endl;
    }
    file << std::endl;

    if (r > 1) {
        for (unsigned int i = 0; i < pomdp->r && i < r; i++) {
            for (unsigned int s = 0; s < pomdp->n; s++) {
                file << "let B[" << (i + 1) << ", " << (s + 1) << "] := " << pomdp->B[i * pomdp->n + s] << ";" << std::endl;
            }
        }
    } else if (r == 1) {
        for (unsigned int s = 0; s < pomdp->n; s++) {
            file << "let b0[" << (s + 1) << "] := " << pomdp->B[0 * pomdp->n + s] << ";" << std::endl;
        }
    }
    file << std::endl;

    file << "let gamma := " << pomdp->gamma << ";" << std::endl;
    file << std::endl;

    for (unsigned int s = 0; s < pomdp->n; s++) {
        for (unsigned int a = 0; a < pomdp->m; a++) {
            for (unsigned int i = 0; i < pomdp->ns; i++) {
                int sp = pomdp->S[s * pomdp->m * pomdp->ns + a * pomdp->ns + i];
                if (sp < 0) {
                    break;
                }

                file << "let T[" << (s + 1) << ", " << (a + 1) << ", " << (sp + 1) << "] := ";
                file << pomdp->T[s * pomdp->m * pomdp->ns + a * pomdp->ns + i] << ";" << std::endl;
            }
        }
    }
    file << std::endl;

    for (unsigned int a = 0; a < pomdp->m; a++) {
        for (unsigned int sp = 0; sp < pomdp->n; sp++) {
            for (unsigned int o = 0; o < pomdp->z; o++) {
                file << "let O[" << (a + 1) << ", " << (sp + 1) << ", " << (o + 1) << "] := ";
                file << pomdp->O[a * pomdp->n * pomdp->z + sp * pomdp->z + o] << ";" << std::endl;
            }
        }
    }
    file << std::endl;

    for (unsigned int s = 0; s < pomdp->n; s++) {
        for (unsigned int a = 0; a < pomdp->m; a++) {

            file << "let R[" << (s + 1) << ", " << (a + 1) << "] := ";
            file << pomdp->R[s * pomdp->m + a] << ";" << std::endl;
        }
    }
    file << std::endl;

    file.close();

    return NOVA_SUCCESS;
}


int pomdp_ampl_parse_solver_output(const POMDP *pomdp, unsigned int k,
        float *policy, float *psi, float *eta, float *V, std::string &solverOutput)
{
    if (k == 0) {
        fprintf(stderr, "Error[pomdp_ampl_parse_solver_output]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Set the default values to 0.0. Not all of the values need to be set because of this.
    if (policy != nullptr) {
        for (unsigned int i = 0; i < k * pomdp->m * pomdp->z * k; i++) {
            policy[i] = 0.0f; 
        }
    }
    if (psi != nullptr) {
        for (unsigned int i = 0; i < k * pomdp->m; i++) {
            psi[i] = 0.0f;
        }
    }
    if (eta != nullptr) {
        for (unsigned int i = 0; i < k * pomdp->m * pomdp->z * k; i++) {
            eta[i] = 0.0f; 
        }
    }
    if (V != nullptr) {
        for (unsigned int i = 0; i < k * pomdp->n; i++) {
            V[i] = 0.0f;
        }
    }

    // Go through every line in the output and parse the result of the solver.
    // Importantly, we assume the solver's output is of the form:
    // <x> <a> <o> <xp> <probability>.
    std::stringstream stream(solverOutput);
    std::string line;

    while (std::getline(stream, line, '\n')) {
        // Get the relevant data from the line.
        std::string data[5];
        unsigned int counter = 0;
        bool newSpace = true;

        for (unsigned int i = 0; i < line.length() && counter < 5; i++) {
            if (line[i] == ' ' && newSpace) {
                counter++;
                newSpace = false;
            } else if (line[i] != ' ' && !newSpace) {
                newSpace = true;
            }

            if (line[i] != ' ') {
                data[counter] += line[i];
            }
        }

        // All data elements need to store some kind of data.
        if (counter == 5 && data[0] == "policy" && policy != nullptr) {
            // Read the raw data as 'policy' for now, which contains psi and eta.
            int x = std::atoi(data[1].c_str());
            int a = std::atoi(data[2].c_str());
            int o = std::atoi(data[3].c_str());
            int xp = std::atoi(data[4].c_str());
            float probability = std::atof(data[5].c_str());

            if (x < 0 || x >= k || a < 0 || a >= pomdp->m ||
                    o < 0 || o >= pomdp->z || xp < 0 || xp >= k ||
                    probability < 0.0f || probability > 1.0f) {
                fprintf(stderr, "Error[pomdp_ampl_parse_solver_output]: %s\n",
                                "Failed to parse the policy.");
                return NOVA_ERROR_INVALID_DATA;
            } else {
                policy[x * pomdp->m * pomdp->z * k +
                       a * pomdp->z * k + o * k + xp] = probability; 
            }
        } else if (counter == 3 && data[0] == "psi" && psi != nullptr) {
            // Read the raw data as 'psi' for now.
            int x = std::atoi(data[1].c_str());
            int a = std::atoi(data[2].c_str());
            float probability = std::atof(data[4].c_str());

            if (x < 0 || x >= k || a < 0 || a >= pomdp->m ||
                    probability < 0.0f || probability > 1.0f) {
                fprintf(stderr, "Error[pomdp_ampl_parse_solver_output]: %s\n",
                                "Failed to parse eta.");
                return NOVA_ERROR_INVALID_DATA;
            } else {
                psi[x * pomdp->m + a] = probability; 
            }
        } else if (counter == 5 && data[0] == "eta" && eta != nullptr) {
            // Read the raw data as 'eta' for now.
            int x = std::atoi(data[1].c_str());
            int a = std::atoi(data[2].c_str());
            int o = std::atoi(data[3].c_str());
            int xp = std::atoi(data[4].c_str());
            float probability = std::atof(data[5].c_str());

            if (x < 0 || x >= k || a < 0 || a >= pomdp->m ||
                    o < 0 || o >= pomdp->z || xp < 0 || xp >= k ||
                    probability < 0.0f || probability > 1.0f) {
                fprintf(stderr, "Error[pomdp_ampl_parse_solver_output]: %s\n",
                                "Failed to parse eta.");
                return NOVA_ERROR_INVALID_DATA;
            } else {
                eta[x * pomdp->m * pomdp->z * k +
                    a * pomdp->z * k + o * k + xp] = probability; 
            }
        } else if (counter == 3 && data[0] == "value" && V != nullptr) {
            // Read the raw data as 'V' for now.
            int x = std::atoi(data[1].c_str());
            int s = std::atoi(data[2].c_str());
            float value = std::atof(data[3].c_str());

            if (x < 0 || x >= k || s < 0 || s >= pomdp->n) {
                fprintf(stderr, "Error[pomdp_ampl_parse_solver_output]: %s\n",
                                "Failed to parse the value.");
                return NOVA_ERROR_INVALID_DATA;
            } else {
                V[x * pomdp->n + s] = value;
            }
        }
    }

    return NOVA_SUCCESS;
}

}; // namespace nova


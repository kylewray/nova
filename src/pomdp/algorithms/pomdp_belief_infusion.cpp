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


#include <nova/pomdp/algorithms/pomdp_belief_infusion.h>

#include <nova/pomdp/utilities/pomdp_ampl.h>
#include <nova/error_codes.h>
#include <nova/constants.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>

#include <stdexcept>
#include <fstream>
#include <string>
#include <sstream>

namespace nova {

int pomdp_bi_save_model_file(const POMDP *pomdp, POMDPBeliefInfusion *bi, const char *filename)
{
    std::string pathAndFilename(bi->path);
    pathAndFilename += "/";
    pathAndFilename += filename;

    std::ofstream file(pathAndFilename, std::ofstream::out);
    if (!file.is_open()) {
        fprintf(stderr, "Error[pomdp_bi_save_model_file]: %s\n", "Failed to save the model file.");
        return NOVA_ERROR_FAILED_TO_OPEN_FILE;
    }

    file << "model;" << std::endl;
    file << std::endl;

    file << "param NUM_STATES;" << std::endl;
    file << "param NUM_ACTIONS;" << std::endl;
    file << "param NUM_OBSERVATIONS;" << std::endl;
    file << "param NUM_NODES;" << std::endl;
    file << "param NUM_BELIEFS;" << std::endl;
    file << std::endl;

    file << "param gamma default 0.95, >= 0.0, <= 1.0;" << std::endl;
    file << "param B {i in 1..NUM_BELIEFS, s in 1..NUM_STATES} default 0.0, >= 0.0, <= 1.0;" << std::endl;
    file << "param lambda {q in 1..NUM_NODES, i in 1..NUM_BELIEFS} default 0.0, >= 0.0, <= 1.0;" << std::endl;
    file << std::endl;

    file << "param T {s in 1..NUM_STATES, a in 1..NUM_ACTIONS, sp in 1..NUM_STATES} ";
    file << "default 0.0, >= 0.0, <= 1.0;" << std::endl;
    file << "param O {a in 1..NUM_ACTIONS, s in 1..NUM_STATES, o in 1..NUM_OBSERVATIONS} ";
    file << "default 0.0, >= 0.0, <= 1.0;" << std::endl;
    file << "param R {s in 1..NUM_STATES, a in 1..NUM_ACTIONS} default 0.0;" << std::endl;
    file << std::endl;

    file << "var V {1..NUM_NODES, 1..NUM_STATES};" << std::endl;
    file << "var policy {1..NUM_NODES, 1..NUM_ACTIONS, 1..NUM_OBSERVATIONS, 1..NUM_NODES} >= 0.0;" << std::endl;
    file << std::endl;

    file << "maximize Value:" << std::endl;
    file << "   sum {s in 1..NUM_STATES} B[1, s] * V[1, s];" << std::endl;
    //file << "   sum {i in 1..NUM_BELIEFS, s in 1..NUM_STATES} lambda[1, i] * B[i, s] * V[i, s];" << std::endl;
    file << std::endl;

    file << "subject to Bellman_Constraint_V {q in 1..NUM_NODES, s in 1..NUM_STATES}:" << std::endl;
    file << "   V[q, s] = sum {a in 1..NUM_ACTIONS} ((sum {qp in 1..NUM_NODES} policy[q, a, 1, qp]) * R[s, a] + ";
    file << "gamma * (1 - sum {i in 1..NUM_BELIEFS} lambda[q, i]) * sum {sp in 1..NUM_STATES} ";
    file << "(T[s, a, sp] * sum {o in 1..NUM_OBSERVATIONS} (O[a, sp, o] * ";
    file << "sum {qp in 1..NUM_NODES} (policy[q, a, o, qp] * V[qp, sp]))) + ";
    file << "gamma * sum {i in 1..NUM_BELIEFS} lambda[q, i] * (sum {sp in 1..NUM_STATES} (B[i, sp] * V[i, sp]))";
    file << ");" << std::endl;
    file << std::endl;

    file << "subject to Probability_Constraint_Normalization ";
    file << "{q in 1..NUM_NODES, o in 1..NUM_OBSERVATIONS}:" << std::endl;
    file << "   sum {qp in 1..NUM_NODES, a in 1..NUM_ACTIONS} policy[q, a, o, qp] = 1.0;" << std::endl;
    file << std::endl;

    file << "subject to Probability_Constraint_Action_Probabilities ";
    file << "{q in 1..NUM_NODES, a in 1..NUM_ACTIONS, o in 1..NUM_OBSERVATIONS}:" << std::endl;
    file << "   sum {qp in 1..NUM_NODES} policy[q, a, o, qp] = sum {qp in 1..NUM_NODES} policy[q, a, 1, qp];" << std::endl;
    file << std::endl;

    file.close();

    return NOVA_SUCCESS;
}


int pomdp_bi_save_data_file_beliefs(const POMDP *pomdp, POMDPBeliefInfusion *bi, const char *filename)
{
    std::string pathAndFilename(bi->path);
    pathAndFilename += "/";
    pathAndFilename += filename;

    std::ofstream file(pathAndFilename, std::ofstream::out | std::ofstream::app);
    if (!file.is_open()) {
        return NOVA_ERROR_FAILED_TO_OPEN_FILE;
    }

    file << "let NUM_BELIEFS := " << bi->r << ";" << std::endl;
    file << std::endl;

    for (unsigned int i = 0; i < bi->r; i++) {
        for (unsigned int s = 0; s < pomdp->n; s++) {
            file << "let B[" << (i + 1) << ", " << (s + 1) << "] := " << bi->B[i * pomdp->n + s] << ";" << std::endl;
        }
    }
    file << std::endl;

    for (unsigned int q = 0; q < bi->k; q++) {
        for (unsigned int i = 0; i < bi->r; i++) {
            file << "let lambda[" << (q + 1) << ", " << (i + 1) << "] := " << bi->lambda[q * bi->r + i] << ";" << std::endl;
        }
    }
    file << std::endl;

    file.close();

    return NOVA_SUCCESS;
}


int pomdp_bi_explore_beliefs(const POMDP *pomdp, POMDPBeliefInfusion *bi)
{
    // Special: The initial belief is only weighted for the first controller node since this
    // comes from the actual POMDP description in most cases. This makes the generalization
    // possible in math, as well as simplifies the objective function itself (only using b0).
    memcpy(bi->B, pomdp->B, pomdp->n * sizeof(float));
    bi->lambda[0 * bi->r + 0] = 1.0f;

    // TODO: For the time being, copy the values from the POMDP... Assume we did the
    // expansion/selection beforehand.
    for (unsigned int i = 1; i < bi->r; i++) {
        if (i >= pomdp->r) {
            break;
        }

        memcpy(&bi->B[i * pomdp->n + 0], &pomdp->B[i * pomdp->n + 0], pomdp->n * sizeof(float));
    }

    // TODO: We want this to basically be the probability of reaching the belief from
    // the "controller node" (only makes sense for initial r controller nodes...).
    // Anyway, for now just assign them all uniform probability that sums to a value < 1.0.
    for (unsigned int q = 1; q < bi->k; q++) {
        for (unsigned int i = 0; i < bi->r; i++) {
            bi->lambda[q * bi->r + i] = 0.3f / (float)(bi->r - 1);
        }
    }

    return NOVA_SUCCESS;
}


int pomdp_bi_execute_solver(const POMDP *pomdp, POMDPBeliefInfusion *bi, std::string &result)
{
    result = "";

    // Attempt to spawn a process for executing the NLP solver.
    char buffer[512];

    FILE *pipe = popen(bi->command, "r");
    if (pipe == nullptr) {
        fprintf(stderr, "Error[pomdp_bi_execute_solver]: %s\n",
                        "Failed to open the process for the solver command.");
        return NOVA_ERROR_EXECUTING_COMMAND;
    }

    // Attempt to execute the NLP solver command. Store the stdout (only) for later parsing.
    try {
        while (!feof(pipe)) {
            if (fgets(buffer, 511, pipe) != nullptr) {
                result += buffer;
            }
        }
    } catch (std::exception &e) {
        pclose(pipe);
        fprintf(stderr, "Error[pomdp_bi_execute_solver]: %s\n",
                        "Failed to execute the solver command via the process.");
        return NOVA_ERROR_EXECUTING_COMMAND;
    }

    pclose(pipe);

    return NOVA_SUCCESS;
}


int pomdp_bi_parse_solver_output(const POMDP *pomdp, POMDPBeliefInfusion *bi, std::string &solverOutput)
{
    // Set the default values to 0.0. Not all of the values need to be set because of this.
    for (unsigned int q = 0; q < bi->k; q++) {
        for (unsigned int a = 0; a < pomdp->m; a++) {
            for (unsigned int o = 0; o < pomdp->z; o++) {
                for (unsigned int qp = 0; qp < bi->k; qp++) {
                    bi->policy[q * pomdp->m * pomdp->z * bi->k +
                               a * pomdp->z * bi->k + o * bi->k + qp] = 0.0f; 
                }
            }
        }
    }

    // Go through every line in the output and parse the result of the solver.
    // Importantly, we assume the solver's output is of the form:
    // <q> <a> <o> <qp> <probability>.
    std::stringstream stream(solverOutput);
    std::string line;

    while (std::getline(stream, line, '\n')) {
        // Get the relevant data from the line.
        std::string data[5];
        unsigned int counter = 0;
        bool newSpace = true;

        for (unsigned int i = 0; i < line.length(); i++) {
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
        if (counter != 4) {
            continue;
        }

        // Read the raw data as 'policy' for now, which contains psi and eta.
        int q = std::atoi(data[0].c_str());
        int a = std::atoi(data[1].c_str());
        int o = std::atoi(data[2].c_str());
        int qp = std::atoi(data[3].c_str());
        float probability = std::atof(data[4].c_str());

        if (q < 0 || q >= bi->k || a < 0 || a >= pomdp->m ||
                o < 0 || o >= pomdp->z || qp < 0 || qp >= bi->k ||
                probability < 0.0f || probability > 1.0f) {
            fprintf(stderr, "Error[pomdp_bi_parse_solver_output]: %s\n",
                            "Failed to parse the policy.");
            return NOVA_ERROR_INVALID_DATA;
        } else {
            bi->policy[q * pomdp->m * pomdp->z * bi->k +
                       a * pomdp->z * bi->k + o * bi->k + qp] = probability; 
        }
    }

    return NOVA_SUCCESS;
}


int pomdp_bi_execute(const POMDP *pomdp, POMDPBeliefInfusion *bi, POMDPStochasticFSC *policy)
{
    // Ensure the data is valid.
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            pomdp->Z == nullptr || pomdp->B == nullptr ||
            pomdp->gamma < 0.0f || pomdp->gamma > 1.0f || pomdp->horizon < 1 ||
            bi == nullptr || bi->k == 0 || policy == nullptr) {
        fprintf(stderr, "Error[pomdp_bi_execute]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    int result = pomdp_bi_initialize(pomdp, bi);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_bi_execute]: %s\n", "Failed to initialize  variables.");
        return result;
    }

    result = pomdp_bi_update(pomdp, bi);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_bi_execute]: %s\n", "Failed to perform belief-infused NLP update step.");
        return result;
    }

    result = pomdp_bi_get_policy(pomdp, bi, policy);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_bi_execute]: %s\n", "Failed to get the policy.");
    }

    result = pomdp_bi_uninitialize(pomdp, bi);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_bi_execute]: %s\n", "Failed to uninitialize the  variables.");
        return result;
    }

    return NOVA_SUCCESS;
}


int pomdp_bi_initialize(const POMDP *pomdp, POMDPBeliefInfusion *bi)
{
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            pomdp->Z == nullptr || pomdp->B == nullptr ||
            pomdp->gamma < 0.0f || pomdp->gamma > 1.0f || pomdp->horizon < 1 ||
            bi == nullptr || bi->k == 0) {
        fprintf(stderr, "Error[pomdp_bi_initialize]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Create the variables that change over iteration (i.e., not the path and command).
    bi->policy = new float[bi->k * pomdp->m * pomdp->z * bi->k];
    for (unsigned int i = 0; i < bi->k * pomdp->m * pomdp->z * bi->k; i++) {
        bi->policy[i] = 0.0f;
    }
    bi->B = new float[bi->k * pomdp->n];
    for (unsigned int i = 0; i < bi->k * pomdp->n; i++) {
        bi->B[i] = 0.0f;
    }
    bi->lambda = new float[bi->k * bi->r];
    for (unsigned int i = 0; i < bi->k * bi->r; i++) {
        bi->lambda[i] = 0.0f;
    }

    // Create the model and data files. Save them for solving.
    int result = pomdp_bi_save_model_file(pomdp, bi, "nova_bi_ampl.mod");
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_bi_initialize]: %s\n", "Failed to save the model file.");
        return NOVA_ERROR_FAILED_TO_OPEN_FILE;
    }

    result = pomdp_ampl_save_data_file(pomdp, bi->k, 0, bi->path, "nova_bi_ampl.dat");
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_bi_initialize]: %s\n", "Failed to save the data file.");
        return NOVA_ERROR_FAILED_TO_OPEN_FILE;
    }

    return NOVA_SUCCESS;
}


int pomdp_bi_update(const POMDP *pomdp, POMDPBeliefInfusion *bi)
{
    // Ensure the data is valid.
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            pomdp->Z == nullptr || pomdp->B == nullptr ||
            pomdp->gamma < 0.0f || pomdp->gamma > 1.0f || pomdp->horizon < 1 ||
            bi == nullptr || bi->k == 0) {
        fprintf(stderr, "Error[pomdp_bi_update]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // First we explore to find a few important beliefs that should be considered during solving.
    // Once these are found, we append them to the AMPL data file.
    int result = pomdp_bi_explore_beliefs(pomdp, bi);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_bi_update]: %s\n", "Failed to explore beliefs.");
        return NOVA_ERROR_EXPLORING_BELIEFS;
    }

    result = pomdp_bi_save_data_file_beliefs(pomdp, bi, "nova_bi_ampl.dat");
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_bi_update]: %s\n", "Failed to save the data file extras.");
        return NOVA_ERROR_FAILED_TO_OPEN_FILE;
    }

    // Now that the file is finished, we can execute the solver and parse the result to store as psi and eta.
    std::string solverOutput = "";
    result = pomdp_bi_execute_solver(pomdp, bi, solverOutput);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_bi_update]: %s\n",
                        "Failed to execute the solver.");
        return NOVA_ERROR_EXECUTING_COMMAND;
    }

    result = pomdp_bi_parse_solver_output(pomdp, bi, solverOutput);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_bi_update]: %s\n",
                        "Failed to parse the result to obtain the policy.");
        return NOVA_ERROR_POLICY_CREATION;
    }

    return NOVA_SUCCESS;
}


int pomdp_bi_get_policy(const POMDP *pomdp, POMDPBeliefInfusion *bi, POMDPStochasticFSC *policy)
{
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->m == 0 || pomdp->r == 0 ||
            bi == nullptr || policy == nullptr) {
        fprintf(stderr, "Error[pomdp_bi_get_policy]: %s\n",
                        "Invalid arguments. Policy must be undefined.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Initialize the policy. Importantly, this allocates allocates memory. Then copy the policy.
    int result = pomdp_stochastic_fsc_initialize(policy, bi->k, pomdp->m, pomdp->z);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_bi_get_policy]: %s\n", "Could not create the policy.");
        return NOVA_ERROR_POLICY_CREATION;
    }

    // Reset psi and compute it by summing over qp with observation index 0 (arbitrary).
    for (unsigned int q = 0; q < bi->k; q++) {
        for (unsigned int a = 0; a < pomdp->m; a++) {
            policy->psi[q * pomdp->m + a] = 0.0f;
            for (unsigned int qp = 0; qp < bi->k; qp++) {
                policy->psi[q * pomdp->m + a] += bi->policy[q * pomdp->m * pomdp->z * bi->k +
                                                             a * pomdp->z * bi->k + 0 * bi->k + qp]; 
            }
        }
    }

    // For eta, first copy the entire policy, then normalize. The math works out that this is
    // equivalent to the original eta.
    memcpy(policy->eta, bi->policy, bi->k * pomdp->m * pomdp->z * bi->k * sizeof(float));
    for (unsigned int q = 0; q < bi->k; q++) {
        for (unsigned int a = 0; a < pomdp->m; a++) {
            for (unsigned int o = 0; o < pomdp->z; o++) {
                float prActionGivenControllerNode = 0.0f;
                for (unsigned int qp = 0; qp < bi->k; qp++) {
                    // Note: The observation is again zero because we are essentially normalizing
                    // by the probability the action is selected.
                    prActionGivenControllerNode += policy->eta[q * pomdp->m * pomdp->z * bi->k +
                                                               a * pomdp->z * bi->k + 0 * bi->k + qp]; 
                }

                // Note: If the probability of taking this action is zero, this is going to
                // divide by zero. Normally, this would be invalid; however, recall the stochastic
                // process in the stochastic FSC. First it randomly selects an action, then transitions
                // after an observation. Thus, if the action is never taken, then the state transition
                // does not matter at all; it will never be used. Thus, we can safely assign these to
                // zero to prevent the needless nan.

                for (unsigned int qp = 0; qp < bi->k; qp++) {
                    if (prActionGivenControllerNode > 0.0f && prActionGivenControllerNode) {
                        policy->eta[q * pomdp->m * pomdp->z * bi->k +
                                    a * pomdp->z * bi->k + o * bi->k + qp] /= prActionGivenControllerNode;
                    } else {
                        policy->eta[q * pomdp->m * pomdp->z * bi->k +
                                    a * pomdp->z * bi->k + o * bi->k + qp] = 0.0f;
                    }
                }
            }
        }
    }

    return NOVA_SUCCESS;
}


int pomdp_bi_uninitialize(const POMDP *pomdp, POMDPBeliefInfusion *bi)
{
    if (pomdp == nullptr || bi == nullptr) {
        fprintf(stderr, "Error[pomdp_bi_uninitialize]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Note: Only free memory of the variables that change
    // during execution (i.e., not path or command).

    if (bi->policy != nullptr) {
        delete [] bi->policy;
    }
    bi->policy = nullptr;

    return NOVA_SUCCESS;
}

}; // namespace nova



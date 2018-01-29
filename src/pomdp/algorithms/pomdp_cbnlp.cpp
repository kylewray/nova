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


#include <nova/pomdp/algorithms/pomdp_cbnlp.h>

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

#include <iostream>

namespace nova {

int pomdp_cbnlp_save_model_file(const POMDP *pomdp, POMDPCBNLP *cbnlp, const char *filename)
{
    std::string pathAndFilename(cbnlp->path);
    pathAndFilename += "/";
    pathAndFilename += filename;

    std::ofstream file(pathAndFilename, std::ofstream::out);
    if (!file.is_open()) {
        fprintf(stderr, "Error[pomdp_cbnlp_save_model_file]: %s\n", "Failed to save the model file.");
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
    file << "param lambda default 0.0, >= 0.0, <= 1.0;" << std::endl;
    file << std::endl;

    file << "param T {s in 1..NUM_STATES, a in 1..NUM_ACTIONS, sp in 1..NUM_STATES} ";
    file << "default 0.0, >= 0.0, <= 1.0;" << std::endl;
    file << "param O {a in 1..NUM_ACTIONS, s in 1..NUM_STATES, o in 1..NUM_OBSERVATIONS} ";
    file << "default 0.0, >= 0.0, <= 1.0;" << std::endl;
    file << "param R {s in 1..NUM_STATES, a in 1..NUM_ACTIONS} default 0.0;" << std::endl;
    file << std::endl;

    file << "var V {1..(NUM_NODES + NUM_BELIEFS), 1..NUM_STATES};" << std::endl;
    file << "var psi {1..(NUM_NODES + NUM_BELIEFS), 1..NUM_ACTIONS} >= 0.0;" << std::endl;
    file << "var eta {1..(NUM_NODES + NUM_BELIEFS), 1..NUM_ACTIONS, 1..NUM_OBSERVATIONS, ";
    file << "1..NUM_NODES} >= 0.0;" << std::endl;
    file << std::endl;

    file << "maximize Value:" << std::endl;
    file << "   sum {s in 1..NUM_STATES} B[1, s] * V[1, s];" << std::endl;
    file << std::endl;

    file << "subject to Bellman_Constraint_V_Nodes {x in 1..NUM_NODES, s in 1..NUM_STATES}:" << std::endl;
    file << "  V[x, s] = sum {a in 1..NUM_ACTIONS} (psi[x, a] * (R[s, a] ";
    file << "+ (gamma * (1.0 - lambda)) * sum {sp in 1..NUM_STATES} (T[s, a, sp] * sum {o in 1..NUM_OBSERVATIONS} ";
    file << "(O[a, sp, o] * sum {xp in 1..NUM_NODES} (eta[x, a, o, xp] * V[xp, sp]))) ";
    file << "+ (gamma * lambda / NUM_BELIEFS) * sum {xp in (NUM_NODES + 1)..(NUM_NODES + NUM_BELIEFS), ";
    file << "sp in 1..NUM_STATES} (B[xp - NUM_NODES, sp] * V[xp, sp])));" << std::endl;
    file << std::endl;

    file << "subject to Bellman_Constraint_V_Beliefs {x in (NUM_NODES + 1)..(NUM_NODES + NUM_BELIEFS), ";
    file << "s in 1..NUM_STATES}:" << std::endl;
    file << "  V[x, s] = sum {a in 1..NUM_ACTIONS} (psi[x, a] * (R[s, a] ";
    file << "+ gamma * sum {sp in 1..NUM_STATES} (T[s, a, sp] * sum {o in 1..NUM_OBSERVATIONS} ";
    file << "(O[a, sp, o] * sum {xp in 1..NUM_NODES} (eta[x, a, o, xp] * V[xp, sp])))));" << std::endl;
    file << std::endl;

    file << "subject to Probability_Constraint_Normalization_Psi {x in 1..(NUM_NODES + NUM_BELIEFS)}:" << std::endl;
    file << "  sum {a in 1..NUM_ACTIONS} psi[x, a] = 1.0;" << std::endl;
    file << std::endl;

    file << "subject to Probability_Constraint_Normalization_Eta {x in 1..(NUM_NODES + NUM_BELIEFS), ";
    file << "a in 1..NUM_ACTIONS, o in 1..NUM_OBSERVATIONS}:" << std::endl;
    file << "  sum {xp in 1..NUM_NODES} eta[x, a, o, xp] = 1.0;" << std::endl;
    file << std::endl;

    /*
    file << "subject to Probability_Constraint_Cannot_Choose_To_Go_To_Beliefs {x in 1..(NUM_NODES + NUM_BELIEFS), ";
    file << "a in 1..NUM_ACTIONS, o in 1..NUM_OBSERVATIONS}:" << std::endl;
    file << "  sum {xp in (NUM_NODES + 1)..(NUM_NODES + NUM_BELIEFS)} policy[x, a, o, xp] = 0.0;" << std::endl;
    file << std::endl;
    //*/

    /*
    file << "subject to Probability_Constraint_Ignore_Action_Observation {x in (NUM_NODES + 1)..(NUM_NODES + NUM_BELIEFS), ";
    file << "a in 2..NUM_ACTIONS, o in 2..NUM_OBSERVATIONS, xp in 1..(NUM_NODES + NUM_BELIEFS)}:" << std::endl;
    file << "  policy[x, a, o, xp] = policy[x, 1, 1, xp];" << std::endl;
    file << std::endl;
    //*/

    file.close();

    return NOVA_SUCCESS;
}


int pomdp_cbnlp_save_data_file_beliefs(const POMDP *pomdp, POMDPCBNLP *cbnlp, const char *filename)
{
    std::string pathAndFilename(cbnlp->path);
    pathAndFilename += "/";
    pathAndFilename += filename;

    std::ofstream file(pathAndFilename, std::ofstream::out | std::ofstream::app);
    if (!file.is_open()) {
        return NOVA_ERROR_FAILED_TO_OPEN_FILE;
    }

    file << "let NUM_BELIEFS := " << cbnlp->r << ";" << std::endl;
    file << std::endl;

    for (unsigned int i = 0; i < cbnlp->r; i++) {
        for (unsigned int s = 0; s < pomdp->n; s++) {
            file << "let B[" << (i + 1) << ", " << (s + 1) << "] := " << cbnlp->B[i * pomdp->n + s] << ";" << std::endl;
        }
    }
    file << std::endl;

    file << "let lambda := " << cbnlp->lmbd << ";" << std::endl;
    file << std::endl;

    file.close();

    return NOVA_SUCCESS;
}


int pomdp_cbnlp_explore_beliefs(const POMDP *pomdp, POMDPCBNLP *cbnlp)
{
    // Special: The initial belief is only weighted for the first controller node since this
    // comes from the actual POMDP description in most cases. This makes the generalization
    // possible in math, as well as simplifies the objective function itself (only using b0).
    /*
    for (unsigned int i = 0; i < pomdp->rz; i++) {
        int s = pomdp->Z[0 * pomdp->rz + i];
        if (s < 0) {
            break;
        }
        cbnlp->B[0 * pomdp->n + s] = pomdp->B[0 * pomdp->rz + i];
    }
    cbnlp->lmbd[0 * cbnlp->r + 0] = 1.0f;
    */

    // TODO: For the time being, copy the values from the POMDP... Assume we did the
    // expansion/selection beforehand.
    //for (unsigned int i = 1; i < cbnlp->r; i++) { // Uncomment if you uncomment above!
    for (unsigned int i = 0; i < cbnlp->r; i++) {
        if (i >= pomdp->r) {
            break;
        }

        for (unsigned int j = 0; j < pomdp->rz; j++) {
            int s = pomdp->Z[i * pomdp->rz + j];
            if (s < 0) {
                break;
            }
            cbnlp->B[i * pomdp->n + s] = pomdp->B[i * pomdp->rz + j];
        }
    }

    // TODO: Assume this is given. Don't assign here.
    //cbnlp->lmbd = 0.3f;

    return NOVA_SUCCESS;
}


int pomdp_cbnlp_execute_solver(const POMDP *pomdp, POMDPCBNLP *cbnlp, std::string &result)
{
    result = "";

    // Attempt to spawn a process for executing the NLP solver.
    char buffer[512];

    FILE *pipe = popen(cbnlp->command, "r");
    if (pipe == nullptr) {
        fprintf(stderr, "Error[pomdp_cbnlp_execute_solver]: %s\n",
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
        fprintf(stderr, "Error[pomdp_cbnlp_execute_solver]: %s\n",
                        "Failed to execute the solver command via the process.");
        return NOVA_ERROR_EXECUTING_COMMAND;
    }

    pclose(pipe);

    return NOVA_SUCCESS;
}


int pomdp_cbnlp_execute(const POMDP *pomdp, POMDPCBNLP *cbnlp, POMDPStochasticFSC *policy)
{
    // Ensure the data is valid.
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            pomdp->Z == nullptr || pomdp->B == nullptr ||
            pomdp->gamma < 0.0f || pomdp->gamma > 1.0f || pomdp->horizon < 1 ||
            cbnlp == nullptr || cbnlp->k == 0 || policy == nullptr) {
        fprintf(stderr, "Error[pomdp_cbnlp_execute]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    int result = pomdp_cbnlp_initialize(pomdp, cbnlp);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_cbnlp_execute]: %s\n", "Failed to initialize  variables.");
        return result;
    }

    result = pomdp_cbnlp_update(pomdp, cbnlp);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_cbnlp_execute]: %s\n", "Failed to perform belief-infused NLP update step.");
        return result;
    }

    result = pomdp_cbnlp_get_policy(pomdp, cbnlp, policy);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_cbnlp_execute]: %s\n", "Failed to get the policy.");
    }

    result = pomdp_cbnlp_uninitialize(pomdp, cbnlp);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_cbnlp_execute]: %s\n", "Failed to uninitialize the  variables.");
        return result;
    }

    return NOVA_SUCCESS;
}


int pomdp_cbnlp_initialize(const POMDP *pomdp, POMDPCBNLP *cbnlp)
{
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            pomdp->Z == nullptr || pomdp->B == nullptr ||
            pomdp->gamma < 0.0f || pomdp->gamma > 1.0f || pomdp->horizon < 1 ||
            cbnlp == nullptr || cbnlp->k == 0) {
        fprintf(stderr, "Error[pomdp_cbnlp_initialize]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    unsigned int numTotalNodes = cbnlp->k + cbnlp->r;

    // Create the variables that change over iteration (i.e., not the path and command).
    cbnlp->psi = new float[numTotalNodes * pomdp->m];
    for (unsigned int i = 0; i < numTotalNodes * pomdp->m; i++) {
        cbnlp->psi[i] = 0.0f;
    }
    cbnlp->eta = new float[numTotalNodes * pomdp->m * pomdp->z * numTotalNodes];
    for (unsigned int i = 0; i < numTotalNodes * pomdp->m * pomdp->z * numTotalNodes; i++) {
        cbnlp->eta[i] = 0.0f;
    }
    cbnlp->B = new float[cbnlp->r * pomdp->n];
    for (unsigned int i = 0; i < cbnlp->r * pomdp->n; i++) {
        cbnlp->B[i] = 0.0f;
    }
    cbnlp->V = new float[numTotalNodes * pomdp->n];
    for (unsigned int i = 0; i < numTotalNodes * pomdp->n; i++) {
        cbnlp->V[i] = 0.0f;
    }

    // Create the model and data files. Save them for solving.
    int result = pomdp_cbnlp_save_model_file(pomdp, cbnlp, "nova_cbnlp_ampl.mod");
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_cbnlp_initialize]: %s\n", "Failed to save the model file.");
        return NOVA_ERROR_FAILED_TO_OPEN_FILE;
    }

    result = pomdp_ampl_save_data_file(pomdp, cbnlp->k, 0, cbnlp->path, "nova_cbnlp_ampl.dat");
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_cbnlp_initialize]: %s\n", "Failed to save the data file.");
        return NOVA_ERROR_FAILED_TO_OPEN_FILE;
    }

    return NOVA_SUCCESS;
}


int pomdp_cbnlp_update(const POMDP *pomdp, POMDPCBNLP *cbnlp)
{
    // Ensure the data is valid.
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            pomdp->Z == nullptr || pomdp->B == nullptr ||
            pomdp->gamma < 0.0f || pomdp->gamma > 1.0f || pomdp->horizon < 1 ||
            cbnlp == nullptr || cbnlp->k == 0) {
        fprintf(stderr, "Error[pomdp_cbnlp_update]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // First we explore to find a few important beliefs that should be considered during solving.
    // Once these are found, we append them to the AMPL data file.
    int result = pomdp_cbnlp_explore_beliefs(pomdp, cbnlp);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_cbnlp_update]: %s\n", "Failed to explore beliefs.");
        return NOVA_ERROR_EXPLORING_BELIEFS;
    }

    result = pomdp_cbnlp_save_data_file_beliefs(pomdp, cbnlp, "nova_cbnlp_ampl.dat");
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_cbnlp_update]: %s\n", "Failed to save the data file extras.");
        return NOVA_ERROR_FAILED_TO_OPEN_FILE;
    }

    // Now that the file is finished, we can execute the solver and parse the result to store as psi and eta.
    std::string solverOutput = "";
    result = pomdp_cbnlp_execute_solver(pomdp, cbnlp, solverOutput);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_cbnlp_update]: %s\n",
                        "Failed to execute the solver.");
        return NOVA_ERROR_EXECUTING_COMMAND;
    }

    result = pomdp_ampl_parse_solver_output(pomdp, cbnlp->k + cbnlp->r, nullptr, cbnlp->psi, cbnlp->eta, cbnlp->V, solverOutput);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_cbnlp_update]: %s\n",
                        "Failed to parse the result to obtain the policy.");
        return NOVA_ERROR_POLICY_CREATION;
    }

    return NOVA_SUCCESS;
}


int pomdp_cbnlp_get_policy(const POMDP *pomdp, POMDPCBNLP *cbnlp, POMDPStochasticFSC *policy)
{
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->m == 0 || pomdp->r == 0 ||
            cbnlp == nullptr || policy == nullptr) {
        fprintf(stderr, "Error[pomdp_cbnlp_get_policy]: %s\n",
                        "Invalid arguments. Policy must be undefined.");
        return NOVA_ERROR_INVALID_DATA;
    }

    unsigned int numTotalNodes = cbnlp->k + cbnlp->r;

    // Initialize the policy. Importantly, this allocates allocates memory. Then copy the policy.
    int result = pomdp_stochastic_fsc_initialize(policy, numTotalNodes, pomdp->n, pomdp->m, pomdp->z);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_cbnlp_get_policy]: %s\n", "Could not create the policy.");
        return NOVA_ERROR_POLICY_CREATION;
    }

    // Copy psi and eta over.
    memcpy(policy->psi, cbnlp->psi, numTotalNodes * pomdp->m * sizeof(float));
    memcpy(policy->eta, cbnlp->eta, numTotalNodes * pomdp->m * pomdp->z * numTotalNodes * sizeof(float));

    /* TODO: Delete this...
    // Reset psi and compute it by summing over xp with observation index 0 (arbitrary).
    for (unsigned int x = 0; x < numTotalNodes; x++) {
        for (unsigned int a = 0; a < pomdp->m; a++) {
            policy->psi[x * pomdp->m + a] = 0.0f;
            for (unsigned int xp = 0; xp < numTotalNodes; xp++) {
                policy->psi[x * pomdp->m + a] += cbnlp->policy[x * pomdp->m * pomdp->z * numTotalNodes +
                                                               a * pomdp->z * numTotalNodes +
                                                               0 * numTotalNodes + xp]; 
            }
        }
    }

    // For eta, first copy the entire policy, then normalize. The math works out that this is
    // equivalent to the original eta.
    memcpy(policy->eta, cbnlp->policy, numTotalNodes * pomdp->m * pomdp->z * numTotalNodes * sizeof(float));
    for (unsigned int x = 0; x < numTotalNodes; x++) {
        for (unsigned int a = 0; a < pomdp->m; a++) {
            for (unsigned int o = 0; o < pomdp->z; o++) {
                float prActionGivenControllerNode = 0.0f;
                for (unsigned int xp = 0; xp < numTotalNodes; xp++) {
                    // Note: The observation is again zero because we are essentially normalizing
                    // by the probability the action is selected.
                    prActionGivenControllerNode += policy->eta[x * pomdp->m * pomdp->z * numTotalNodes +
                                                               a * pomdp->z * numTotalNodes +
                                                               0 * numTotalNodes + xp]; 
                }

                // Note: If the probability of taking this action is zero, this is going to
                // divide by zero. Normally, this would be invalid; however, recall the stochastic
                // process in the stochastic FSC. First it randomly selects an action, then transitions
                // after an observation. Thus, if the action is never taken, then the state transition
                // does not matter at all; it will never be used. Thus, we can safely assign these to
                // zero to prevent the needless nan.

                for (unsigned int xp = 0; xp < numTotalNodes; xp++) {
                    if (prActionGivenControllerNode > 0.0f && prActionGivenControllerNode) {
                        policy->eta[x * pomdp->m * pomdp->z * numTotalNodes +
                                    a * pomdp->z * numTotalNodes +
                                    o * numTotalNodes + xp] /= prActionGivenControllerNode;
                    } else {
                        policy->eta[x * pomdp->m * pomdp->z * numTotalNodes +
                                    a * pomdp->z * numTotalNodes +
                                    o * numTotalNodes + xp] = 0.0f;
                    }
                }
            }
        }
    }
    */

    // Lastly, copy the values of each controller node and state pair.
    memcpy(policy->V, cbnlp->V, numTotalNodes * pomdp->n * sizeof(float));

    return NOVA_SUCCESS;
}


int pomdp_cbnlp_uninitialize(const POMDP *pomdp, POMDPCBNLP *cbnlp)
{
    if (pomdp == nullptr || cbnlp == nullptr) {
        fprintf(stderr, "Error[pomdp_cbnlp_uninitialize]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Note: Only free memory of the variables that change
    // during execution (i.e., not path or command).

    if (cbnlp->B != nullptr) {
        delete [] cbnlp->B;
    }
    cbnlp->B = nullptr;

    if (cbnlp->psi != nullptr) {
        delete [] cbnlp->psi;
    }
    cbnlp->psi = nullptr;

    if (cbnlp->eta != nullptr) {
        delete [] cbnlp->eta;
    }
    cbnlp->eta = nullptr;

    if (cbnlp->V != nullptr) {
        delete [] cbnlp->V;
    }
    cbnlp->V = nullptr;

    return NOVA_SUCCESS;
}

}; // namespace nova



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


#include <nova/pomdp/algorithms/pomdp_nlp.h>

#include <nova/pomdp/utilities/pomdp_ampl.h>
#include <nova/error_codes.h>
#include <nova/constants.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <fstream>
#include <string>

namespace nova {

int pomdp_ampl_save_nlp_model_file(const POMDP *pomdp, const char *path)
{
    std::string filename(path);
    filename += "/nova_nlp_ampl.mod";

    std::ofstream file(filename, std::ofstream::out);
    if (!file.is_open()) {
        return NOVA_ERROR_FAILED_TO_OPEN_FILE;
    }

    file << "model;" << std::endl;

    file << "set STATES;" << std::endl;
    file << "set ACTIONS;" << std::endl;
    file << "set OBSERVATIONS;" << std::endl;
    file << "set CONTROLLER_NODES;" << std::endl;

    file << "param q0 {q in CONTROLLER_NODES} default 0.0, >= 0.0, <= 1.0;" << std::endl;
    file << "param b0 {s in STATES} default 0.0, >= 0.0, <= 1.0;" << std::endl;
    file << "param gamma default 0.95, >= 0.0, <= 1.0;" << std::endl;
    file << "param T {s in STATES, a in ACTIONS, sp in STATES} default 0.0, >= 0.0, <= 1.0;" << std::endl;
    file << "param O {a in ACTIONS, s in STATES, o in OBSERVATIONS} default 0.0, >= 0.0, <= 1.0;" << std::endl;
    file << "param R {s in STATES, a in ACTIONS} default 0.0;" << std::endl;

    file << "var V {CONTROLLER_NODES, STATES};" << std::endl;
    file << "var psi {CONTROLLER_NODES, ACTIONS} >= 0.0, <= 1.0;" << std::endl;
    file << "var eta {CONTROLLER_NODES, ACTIONS, OBSERVATIONS, CONTROLLER_NODES} >= 0.0, <= 1.0;" << std::endl;

    file << "maximize Value:" << std::endl;
    file << "    sum {q in CONTROLLER_NODES, s in STATES} q0[q] * b0[s] * V[q, s];" << std::endl;

    file << "subject to Bellman_Constraint_V {q in CONTROLLER_NODES, s in STATES}:" << std::endl;
    file << "    V[q, s] = sum {a in ACTIONS} (psi[q, a] * (R[s, a] + gamma * sum {sp in STATES} ";
    file << "(T[s, a, sp] * sum {o in OBSERVATIONS} (O[a, sp, o] * sum {qp in CONTROLLER_NODES} ";
    file << "(eta[q, a, o, qp] * V[qp, sp])))));" << std::endl;

    file << "subject to Probability_Constraint_Psi_Nonnegative {q in CONTROLLER_NODES, a in ACTIONS}:" << std::endl;
    file << "    psi[q, a] >= 0.0;" << std::endl;

    file << "subject to Probability_Constraint_Psi_Normalization {q in CONTROLLER_NODES}:" << std::endl;
    file << "    sum {a in ACTIONS} psi[q, a] = 1.0;" << std::endl;

    file << "subject to Probability_Constraint_Eta_Nonnegative {q in CONTROLLER_NODES, a in ACTIONS, ";
    file << "o in OBSERVATIONS, qp in CONTROLLER_NODES}:" << std::endl;
    file << "    eta[q, a, o, qp] >= 0.0;" << std::endl;

    file << "subject to Probability_Constraint_Eta_Normalization {q in CONTROLLER_NODES, a in ACTIONS, ";
    file << "o in OBSERVATIONS}:" << std::endl;
    file << "    sum {qp in CONTROLLER_NODES} eta[q, a, o, qp] = 1.0;" << std::endl;

    file.close();

    return NOVA_SUCCESS;
}


int pomdp_nlp_execute_solver(const POMDP *pomdp, POMDPNLP *nlp, std::string &result)
{
    result = "";

    // Attempt to spawn a process for executing the NLP solver.
    char buffer[512];

    FILE *pipe = popen(nlp->command, "r");
    if (pipe == nullptr) {
        fprintf(stderr, "Error[pomdp_nlp_execute_solver]: %s\n",
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
        fprintf(stderr, "Error[pomdp_nlp_execute_solver]: %s\n",
                        "Failed to execute the solver command via the process.");
        return NOVA_ERROR_EXECUTING_COMMAND;
    }

    pclose(pipe);

    return NOVA_SUCCESS;
}


int pomdp_nlp_parse_solver_output(const POMDP *pomdp, POMDPNLP *nlp, std::string &solverOutput)
{
    for (unsigned int q = 0; q < nlp->k; q++) {
        for (unsigned int a = 0; a < pomdp->m; a++) {
            nlp->psi[q * pomdp->m + a] = 0.0f;
        }
    }

    for (unsigned int q = 0; q < nlp->k; q++) {
        for (unsigned int a = 0; a < pomdp->m; a++) {
            for (unsigned int o = 0; o < pomdp->z; o++) {
                for (unsigned int qp = 0; qp < nlp->k; qp++) {
                    nlp->eta[q * pomdp->m * pomdp->z * nlp->k +
                             a * pomdp->z * nlp->k + o * nlp->k + qp] = 0.0f; 
                }
            }
        }
    }

    // TODO
    printf("%s", solverOutput);

    return NOVA_SUCCESS;
}


int pomdp_nlp_execute(const POMDP *pomdp, POMDPNLP *nlp, POMDPStochasticFSC *policy)
{
    // Ensure the data is valid.
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            pomdp->Z == nullptr || pomdp->B == nullptr ||
            pomdp->gamma < 0.0f || pomdp->gamma > 1.0f || pomdp->horizon < 1 ||
            nlp == nullptr || nlp->k == 0 || policy == nullptr) {
        fprintf(stderr, "Error[pomdp_nlp_execute]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    int result = pomdp_nlp_initialize(pomdp, nlp);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_nlp_execute]: %s\n", "Failed to initialize  variables.");
        return result;
    }

    result = pomdp_nlp_update(pomdp, nlp);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_nlp_execute]: %s\n", "Failed to perform NLP update step.");
        return result;
    }

    result = pomdp_nlp_get_policy(pomdp, nlp, policy);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_nlp_execute]: %s\n", "Failed to get the policy.");
    }

    result = pomdp_nlp_uninitialize(pomdp, nlp);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_nlp_execute]: %s\n", "Failed to uninitialize the  variables.");
        return result;
    }

    return NOVA_SUCCESS;
}


int pomdp_nlp_initialize(const POMDP *pomdp, POMDPNLP *nlp)
{
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            pomdp->Z == nullptr || pomdp->B == nullptr ||
            pomdp->gamma < 0.0f || pomdp->gamma > 1.0f || pomdp->horizon < 1 ||
            nlp == nullptr || nlp->k == 0) {
        fprintf(stderr, "Error[pomdp_nlp_initialize]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Create the variables that change over iteration (i.e., not the path and command).
    nlp->psi = new float[nlp->k * pomdp->m];
    nlp->eta = new float[nlp->k * pomdp->m * pomdp->z * nlp->k];

    for (unsigned int i = 0; i < nlp->k * pomdp->m; i++) {
        nlp->psi[i] = 0.0f;
    }
    for (unsigned int i = 0; i < nlp->k * pomdp->m * pomdp->z * nlp->k; i++) {
        nlp->eta[i] = 0.0f;
    }

    // Create the model and data files. Save them for solving.
    int result = pomdp_ampl_save_nlp_model_file(pomdp, nlp->path);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_nlp_initialize]: %s\n", "Failed to save the model file.");
        return NOVA_ERROR_FAILED_TO_OPEN_FILE;
    }

    result = pomdp_ampl_save_data_file(pomdp, nlp->k, nlp->path);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_nlp_initialize]: %s\n", "Failed to save the data file.");
        return NOVA_ERROR_FAILED_TO_OPEN_FILE;
    }

    return NOVA_SUCCESS;
}


int pomdp_nlp_update(const POMDP *pomdp, POMDPNLP *nlp)
{
    // Ensure the data is valid.
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            pomdp->Z == nullptr || pomdp->B == nullptr ||
            pomdp->gamma < 0.0f || pomdp->gamma > 1.0f || pomdp->horizon < 1 ||
            nlp == nullptr || nlp->k == 0) {
        fprintf(stderr, "Error[pomdp_nlp_update]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    std::string solverOutput = "";
    int result = pomdp_nlp_execute_solver(pomdp, nlp, solverOutput);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_nlp_update]: %s\n",
                        "Failed to execute the solver.");
        return NOVA_ERROR_EXECUTING_COMMAND;
    }

    result = pomdp_nlp_parse_solver_output(pomdp, nlp, solverOutput);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_nlp_update]: %s\n",
                        "Failed to parse the result to obtain the policy.");
        return NOVA_ERROR_POLICY_CREATION;
    }

    return NOVA_SUCCESS;
}


int pomdp_nlp_get_policy(const POMDP *pomdp, POMDPNLP *nlp, POMDPStochasticFSC *policy)
{
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->m == 0 || pomdp->r == 0 ||
            nlp == nullptr || policy == nullptr) {
        fprintf(stderr, "Error[pomdp_nlp_get_policy]: %s\n",
                        "Invalid arguments. Policy must be undefined.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Initialize the policy. Importantly, this allocates allocates memory. Then copy the policy.
    int result = pomdp_stochastic_fsc_initialize(policy, nlp->k, pomdp->m, pomdp->z);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_nlp_get_policy]: %s\n", "Could not create the policy.");
        return NOVA_ERROR_POLICY_CREATION;
    }

    memcpy(policy->psi, nlp->psi, nlp->k * pomdp->m * sizeof(float));
    memcpy(policy->eta, nlp->eta, nlp->k * pomdp->m * pomdp->z * nlp->k * sizeof(float));

    return NOVA_SUCCESS;
}


int pomdp_nlp_uninitialize(const POMDP *pomdp, POMDPNLP *nlp)
{
    if (pomdp == nullptr || nlp == nullptr) {
        fprintf(stderr, "Error[pomdp_nlp_uninitialize]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Note: Only free memory of the variables that change
    // during execution (i.e., not path or command).

    if (nlp->psi != nullptr) {
        delete [] nlp->psi;
    }
    nlp->psi = nullptr;

    if (nlp->eta != nullptr) {
        delete [] nlp->eta;
    }
    nlp->eta = nullptr;

    return NOVA_SUCCESS;
}

}; // namespace nova


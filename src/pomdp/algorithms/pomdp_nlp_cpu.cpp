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


#include <nova/pomdp/algorithms/pomdp_nlp_cpu.h>

#include <nova/error_codes.h>
#include <nova/constants.h>

#include <stdio.h>

namespace nova {

int pomdp_nlp_execute_cpu(const POMDP *pomdp, POMDPNLPCPU *nlp, POMDPStochasticFSC *policy)
{
    // Ensure the data is valid.
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            pomdp->Z == nullptr || pomdp->B == nullptr ||
            pomdp->gamma < 0.0f || pomdp->gamma > 1.0f || pomdp->horizon < 1 ||
            nlp == nullptr || nlp->k == 0 || policy == nullptr) {
        fprintf(stderr, "Error[pomdp_nlp_execute_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    int result = pomdp_nlp_initialize_cpu(pomdp, nlp);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_nlp_execute_cpu]: %s\n", "Failed to initialize CPU variables.");
        return result;
    }

    result = pomdp_nlp_update_cpu(pomdp, nlp);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_nlp_execute_cpu]: %s\n", "Failed to perform NLP update step.");
        return result;
    }

    result = pomdp_nlp_get_policy_cpu(pomdp, nlp, policy);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_nlp_execute_cpu]: %s\n", "Failed to get the policy.");
    }

    result = pomdp_nlp_uninitialize_cpu(pomdp, nlp);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_nlp_execute_cpu]: %s\n", "Failed to uninitialize the CPU variables.");
        return result;
    }

    return NOVA_SUCCESS;
}


int pomdp_nlp_initialize_cpu(const POMDP *pomdp, POMDPNLPCPU *nlp)
{
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            pomdp->Z == nullptr || pomdp->B == nullptr ||
            pomdp->gamma < 0.0f || pomdp->gamma > 1.0f || pomdp->horizon < 1 ||
            nlp == nullptr || nlp->k == 0) {
        fprintf(stderr, "Error[pomdp_nlp_initialize_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // TODO: Save the POMDP to the AMPL model file in a helper function.

    return NOVA_SUCCESS;
}


int pomdp_nlp_update_cpu(const POMDP *pomdp, POMDPNLPCPU *nlp)
{
    // Ensure the data is valid.
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            pomdp->Z == nullptr || pomdp->B == nullptr ||
            pomdp->gamma < 0.0f || pomdp->gamma > 1.0f || pomdp->horizon < 1 ||
            nlp == nullptr || nlp->k == 0) {
        fprintf(stderr, "Error[pomdp_nlp_update_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // TODO: Call the NEOS server for solving the AMPL files.

    return NOVA_SUCCESS;
}


int pomdp_nlp_get_policy_cpu(const POMDP *pomdp, POMDPNLPCPU *nlp, POMDPStochasticFSC *policy)
{
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->m == 0 || pomdp->r == 0 ||
            nlp == nullptr || policy == nullptr) {
        fprintf(stderr, "Error[pomdp_nlp_get_policy_cpu]: %s\n",
                        "Invalid arguments. Policy must be undefined.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Initialize the policy, which allocates memory.
    // TODO: Initialize the policy using an FSC initialize
    int result = pomdp_stochastic_fsc_initialize(policy, nlp->k, pomdp->m, pomdp->z);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_nlp_get_policy_cpu]: %s\n", "Could not create the policy.");
        return NOVA_ERROR_POLICY_CREATION;
    }

    // TODO: Use a helper file to read the result of the NEOS server's solver to assign eta and psi.
    // policy->eta = ???
    // policy->psi = ???

    return NOVA_SUCCESS;
}


int pomdp_nlp_uninitialize_cpu(const POMDP *pomdp, POMDPNLPCPU *nlp)
{
    if (pomdp == nullptr || nlp == nullptr) {
        fprintf(stderr, "Error[pomdp_nlp_uninitialize_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    nlp->k = 0;

    return NOVA_SUCCESS;
}

}; // namespace nova


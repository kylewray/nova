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


#include <nova/pomdp/policies/pomdp_stochastic_fsc.h>

#include <nova/error_codes.h>
#include <nova/constants.h>

#include <stdio.h>
#include <cstdlib>

namespace nova {

int pomdp_stochastic_fsc_initialize(POMDPStochasticFSC *policy, unsigned int k, unsigned int m, unsigned int z)
{
    if (policy == nullptr || policy->psi != nullptr || policy->eta != nullptr || k == 0 || m == 0 || z == 0) {
        fprintf(stderr, "Error[pomdp_stochastic_fsc_initialize]: %s\n", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    policy->k = k;
    policy->m = m;
    policy->z = z;

    policy->psi = new float[policy->k * policy->m];
    policy->eta = new float[policy->k * policy->m * policy->z * policy->k];

    for (unsigned int i = 0; i < policy->k * policy->m; i++) {
        policy->psi[i] = 0.0f;
    }

    for (unsigned int i = 0; i < policy->k * policy->m * policy->z * policy->k; i++) {
        policy->eta[i] = 0.0f;
    }

    return NOVA_SUCCESS;
}


int pomdp_stochastic_fsc_random_action(POMDPStochasticFSC *policy, unsigned int q, unsigned int &a)
{
    if (policy == nullptr || q >= policy->k) {
        fprintf(stderr, "Error[pomdp_stochastic_fsc_random_action]: %s\n", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    a = 0;

    double target = (double)rand() / (double)RAND_MAX;
    double current = 0.0f;

    for (unsigned int i = 0; i < policy->m; i++) {
        current += policy->psi[q * policy->m + i];
        if (current >= target) {
            a = i;
            break;
        }
    }

    return NOVA_SUCCESS;
}


int pomdp_stochastic_fsc_random_successor(POMDPStochasticFSC *policy,
    unsigned int q, unsigned int a, unsigned int o, unsigned int &qp)
{
    if (policy == nullptr || q >= policy->k || a >= policy->m || o >= policy->z) {
        fprintf(stderr, "Error[pomdp_stochastic_fsc_random_successor]: %s\n", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    qp = 0;

    double target = (double)rand() / (double)RAND_MAX;
    double current = 0.0f;

    for (unsigned int i = 0; i < policy->k; i++) {
        current += policy->eta[q * policy->m * policy->z * policy->k + a * policy->z * policy->k + o * policy->k + i];
        if (current >= target) {
            qp = i;
            break;
        }
    }

    return NOVA_SUCCESS;
}


int pomdp_stochastic_fsc_uninitialize(POMDPStochasticFSC *policy)
{
    if (policy == nullptr) {
        fprintf(stderr, "Error[pomdp_stochastic_fsc_uninitialize]: %s\n", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    policy->k = 0;
    policy->m = 0;
    policy->z = 0;

    if (policy->psi != nullptr) {
        delete [] policy->psi;
    }
    policy->psi = nullptr;

    if (policy->eta != nullptr) {
        delete [] policy->eta;
    }
    policy->eta = nullptr;

    //delete policy;
    //policy = nullptr;

    return NOVA_SUCCESS;
}

}; // namespace nova


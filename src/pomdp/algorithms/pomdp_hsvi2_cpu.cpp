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


#include <nova/pomdp/algorithms/pomdp_hsvi2_cpu.h>

#include <nova/pomdp/policies/pomdp_alpha_vectors.h>
#include <nova/pomdp/utilities/pomdp_model_cpu.h>

#include <nova/error_codes.h>
#include <nova/constants.h>

#include <stdio.h>
#include <cstring>
#include <cstdlib>
#include <time.h>
#include <algorithm>

namespace nova {

int pomdp_hsvi2_lower_bound_initialize_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2)
{
    return NOVA_SUCCESS;
}


int pomdp_hsvi2_upper_bound_initialize_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2)
{
    return NOVA_SUCCESS;
}


float pomdp_hsvi2_width_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2, float *b)
{
    return 0.0f;
}


unsigned int pomdp_hsvi2_compute_a_star(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2, float *b)
{
    return 0;
}


unsigned int pomdp_hsvi2_compute_o_star(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2, float *b, unsigned int aStar, float excess)
{
    return 0;
}


int pomdp_hsvi2_lower_bound_update_step_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2, float *b)
{
    return NOVA_SUCCESS;
}


int pomdp_hsvi2_upper_bound_update_step_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2, float *b)
{
    return NOVA_SUCCESS;
}


int pomdp_hsvi2_initialize_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2)
{
    if (pomdp == nullptr || pomdp->n == 0 || hsvi2 == nullptr || hsvi2->maxAlphaVectors == 0) {
        fprintf(stderr, "Error[pomdp_hsvi2_initialize_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Reset the current horizon and trials.
    hsvi2->currentTrial = 0;

    // Create the lower and upper bound variables.
    hsvi2->lowerGamma = new float[hsvi2->maxAlphaVectors * pomdp->n];
    hsvi2->lowerPi = new unsigned int[hsvi2->maxAlphaVectors];
    hsvi2->upperGammaB = new float[hsvi2->maxAlphaVectors * pomdp->n];
    hsvi2->upperGammaHVb = new float[hsvi2->maxAlphaVectors];

    for (unsigned int i = 0; i < hsvi2->maxAlphaVectors * pomdp->n; i++) {
        hsvi2->lowerGamma[i] = 0.0f;
        hsvi2->upperGammaB[i] = 0.0f;
    }

    for (unsigned int i = 0; i < hsvi2->maxAlphaVectors; i++) {
        hsvi2->lowerPi[i] = 0;
        hsvi2->upperGammaHVb[i] = 0.0f;
    }

    int result = pomdp_hsvi2_lower_bound_initialize_cpu(pomdp, hsvi2);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_hsvi2_initialize_cpu]: Failed to initialize the lower bound.");
        return result;
    }

    result = pomdp_hsvi2_upper_bound_initialize_cpu(pomdp, hsvi2);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_hsvi2_initialize_cpu]: Failed to initialize the upper bound.");
        return result;
    }

    return NOVA_SUCCESS;
}


int pomdp_hsvi2_execute_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2, POMDPAlphaVectors *policy)
{
    // Ensure the data is valid.
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            pomdp->Z == nullptr || pomdp->B == nullptr ||
            pomdp->gamma < 0.0f || pomdp->gamma > 1.0f || pomdp->horizon < 1 ||
            hsvi2 == nullptr || policy == nullptr) {
        fprintf(stderr, "Error[pomdp_hsvi2_execute_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    int result = pomdp_hsvi2_initialize_cpu(pomdp, hsvi2);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_hsvi2_execute_cpu]: %s\n", "Failed to initialize CPU variables.");
        return result;
    }

    // Iterate until either the algorithm has converged or you have reached the maximal number of trials.
    for (hsvi2->currentTrial = 0;
            hsvi2->currentTrial < hsvi2->trials && result != NOVA_CONVERGED;
            hsvi2->currentTrial++) {

        result = pomdp_hsvi2_update_cpu(pomdp, hsvi2);
        if (result != NOVA_SUCCESS && result != NOVA_CONVERGED) {
            fprintf(stderr, "Error[pomdp_hsvi2_execute_cpu]: %s\n",
                            "Failed to perform trial of HSVI2 on the CPU.");

            unsigned int resultPrime = pomdp_hsvi2_uninitialize_cpu(pomdp, hsvi2);
            if (resultPrime != NOVA_SUCCESS) {
                fprintf(stderr, "Error[pomdp_hsvi2_execute_cpu]: %s\n",
                                "Failed to uninitialize the CPU variables.");
            }

            return result;
        }
    }

    result = pomdp_hsvi2_get_policy_cpu(pomdp, hsvi2, policy);
    if (result != NOVA_SUCCESS && result != NOVA_WARNING_APPROXIMATE_SOLUTION) {
        fprintf(stderr, "Error[pomdp_hsvi2_execute_cpu]: %s\n", "Failed to get the policy.");
    }

    bool approximateSolution = (result == NOVA_WARNING_APPROXIMATE_SOLUTION);

    result = pomdp_hsvi2_uninitialize_cpu(pomdp, hsvi2);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_hsvi2_execute_cpu]: %s\n", "Failed to uninitialize the CPU variables.");
        return result;
    }

    // If this was an approximate solution, return this warning. Otherwise, return success.
    if (approximateSolution) {
        fprintf(stderr, "Warning[ssp_hsvi2_execute_cpu]: %s\n",
                "Approximate solution due to reaching the maximum number of trials.");
        return NOVA_WARNING_APPROXIMATE_SOLUTION;
    }

    return NOVA_SUCCESS;
}


int pomdp_hsvi2_uninitialize_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2)
{
    if (pomdp == nullptr || hsvi2 == nullptr) {
        fprintf(stderr, "Error[pomdp_hsvi2_uninitialize_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Reset the current horizon and number of trials, as well as the sizes of lower/upper Gamma arrays.
    hsvi2->currentTrial = 0;

    hsvi2->lowerGammaSize = 0;
    hsvi2->upperGammaSize = 0;

    // Free the memory for the lower and upper bounds.
    if (hsvi2->lowerGamma != nullptr) {
        delete [] hsvi2->lowerGamma;
    }
    hsvi2->lowerGamma = nullptr;

    if (hsvi2->lowerPi != nullptr) {
        delete [] hsvi2->lowerPi;
    }
    hsvi2->lowerPi = nullptr;

    if (hsvi2->upperGammaB != nullptr) {
        delete [] hsvi2->upperGammaB;
    }
    hsvi2->upperGammaB = nullptr;

    if (hsvi2->upperGammaHVb != nullptr) {
        delete [] hsvi2->upperGammaHVb;
    }
    hsvi2->upperGammaHVb = nullptr;

    return NOVA_SUCCESS;
}


int pomdp_hsvi2_update_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2)
{
    // Ensure the data is valid.
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            pomdp->Z == nullptr || pomdp->B == nullptr ||
            pomdp->gamma < 0.0f || pomdp->gamma > 1.0f || pomdp->horizon < 1 ||
            hsvi2 == nullptr || hsvi2->lowerGamma == nullptr || hsvi2->lowerPi == nullptr ||
            hsvi2->upperGammaB == nullptr || hsvi2->upperGammaHVb == nullptr) {
        fprintf(stderr, "Error[pomdp_hsvi2_update_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Check if we ran out of memory.
    if (hsvi2->lowerGammaSize >= hsvi2->maxAlphaVectors || hsvi2->upperGammaSize >= hsvi2->maxAlphaVectors) {
        fprintf(stderr, "Error[pomdp_hsvi2_update_cpu]: %s\n", "Out of memory in HSVI2 exploration step.");
        return NOVA_ERROR_OUT_OF_MEMORY;
    }

    // Note: This is essentially "explore(b0, epsilon, 0)" from the HSVI2 paper.

    // Create the history of beliefs for post order traversal lower and upper bound updating.
    float *traversalBeliefStack = new float[pomdp->horizon * pomdp->n];
    for (unsigned int i = 0; i < pomdp->horizon * pomdp->n; i++) {
        traversalBeliefStack[i] = 0.0f;
    }

    // Copy the initial belief in the POMDP to begin the search.
    float *b = new float[pomdp->n];
    for (unsigned int i = 0; i < pomdp->rz; i++) {
        int s = pomdp->Z[0 * pomdp->rz + i];
        if (s < 0) {
            break;
        }
        b[s] = pomdp->B[0 * pomdp->rz + i];
    }

    // Iterate until either the width is less than the convergence criterion or
    // the maximum horizon depth is reached.
    float gammaPowerT = 1.0f;
    unsigned int currentHorizon = 0;

    for (currentHorizon = 0; currentHorizon < pomdp->horizon; currentHorizon++) {
        // Check if the width is smaller than the convergence criterion; break if this is the case.
        float excess = pomdp_hsvi2_width_cpu(pomdp, hsvi2, b) - hsvi2->epsilon / gammaPowerT;
        if (excess <= 0.0f) {
            break;
        }

        gammaPowerT *= pomdp->gamma;

        // Select an action according to the forward exploration heuristics.
        unsigned int aStar = pomdp_hsvi2_compute_a_star(pomdp, hsvi2, b);

        // Select an observation according to the forward exploration heuristics.
        unsigned int oStar = pomdp_hsvi2_compute_o_star(pomdp, hsvi2, b, aStar, excess);

        // Push the current belief point on the "traversal belief stack".
        memcpy(&traversalBeliefStack[currentHorizon * pomdp->n], b, pomdp->n * sizeof(float));

        // Transition to the new belief given this action-observation. Note: The update allocates more memory.
        delete [] b;
        b = nullptr;

        pomdp_belief_update_cpu(pomdp, &traversalBeliefStack[currentHorizon * pomdp->n], aStar, oStar, b);
    }

    // In post-order traversal, perform an update on each belief points for both the lower and upper bounds.
    bool ranOutOfMemory = false;

    for (int i = std::max((int)currentHorizon - 1, 0); i >= 0 && !ranOutOfMemory; i--) {
        int lowerResult = pomdp_hsvi2_lower_bound_update_step_cpu(pomdp, hsvi2, &traversalBeliefStack[i * pomdp->n]);
        int upperResult = pomdp_hsvi2_upper_bound_update_step_cpu(pomdp, hsvi2, &traversalBeliefStack[i * pomdp->n]);

        if (lowerResult == NOVA_ERROR_OUT_OF_MEMORY || upperResult == NOVA_ERROR_OUT_OF_MEMORY) {
            ranOutOfMemory = true;
        }
    }

    delete [] traversalBeliefStack;
    delete [] b;
    traversalBeliefStack = nullptr;
    b = nullptr;

    if (ranOutOfMemory) {
        fprintf(stderr, "Error[pomdp_hsvi2_update_cpu]: %s\n", "Out of memory in HSVI2 exploration step.");
        return NOVA_ERROR_OUT_OF_MEMORY;
    }

    return NOVA_SUCCESS;
}


int pomdp_hsvi2_get_policy_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2, POMDPAlphaVectors *policy)
{
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->m == 0 || pomdp->r == 0 ||
            hsvi2 == nullptr || hsvi2->lowerGammaSize == 0 || hsvi2->lowerGamma == nullptr || hsvi2->lowerPi == nullptr ||
            policy == nullptr) {
        fprintf(stderr, "Error[pomdp_hsvi2_get_policy_cpu]: %s\n", "Invalid arguments. Policy must be undefined.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Initialize the policy, which allocates memory.
    int result = pomdp_alpha_vectors_initialize(policy, pomdp->n, pomdp->m, hsvi2->lowerGammaSize);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_hsvi2_get_policy_cpu]: %s\n", "Could not create the policy.");
        return NOVA_ERROR_POLICY_CREATION;
    }

    // Copy the final (or intermediate) result of Gamma and pi to the variables.
    memcpy(policy->Gamma, hsvi2->lowerGamma, hsvi2->lowerGammaSize * pomdp->n * sizeof(float));
    memcpy(policy->pi, hsvi2->lowerPi, hsvi2->lowerGammaSize * sizeof(unsigned int));

    return NOVA_SUCCESS;
}

}; // namespace nova



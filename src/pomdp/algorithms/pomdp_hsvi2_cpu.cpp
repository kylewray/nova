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
#include <nova/pomdp/utilities/pomdp_functions_cpu.h>

#include <nova/mdp/algorithms/mdp_vi_cpu.h>
#include <nova/mdp/policies/mdp_value_function.h>
#include <nova/mdp/utilities/mdp_model_cpu.h>

#include <nova/error_codes.h>
#include <nova/constants.h>

#include <stdio.h>
#include <cstring>
#include <cstdlib>
#include <time.h>
#include <algorithm>
#include <iostream>

namespace nova {

int pomdp_hsvi2_lower_bound_initialize_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2)
{
    if (hsvi2->maxAlphaVectors < pomdp->m) {
        fprintf(stderr, "Error[pomdp_hsvi2_lower_bound_initialize_cpu]: Cannot initialize upper bound. Not enough memory.");
        return NOVA_ERROR_OUT_OF_MEMORY;
    }

    // First, we compute the lowerbound on R = max_a min_s R(s, a) / (1 - gamma).
    float maxminR = NOVA_FLT_MIN;
    for (unsigned int a = 0; a < pomdp->m; a++) {
        float minR = pomdp->R[0 * pomdp->m + a];
        for (unsigned int s = 0; s < pomdp->n; s++) {
            if (pomdp->R[s * pomdp->m + a] < minR) {
                minR = pomdp->R[s * pomdp->m + a];
            }
        }

        if (a == 0 || minR > maxminR) {
            maxminR = minR;
        }
    }

    // Assign alpha^a_0 = maxminR / (1 - gamma) to all initial alpha values.
    for (unsigned int a = 0; a < pomdp->m; a++) {
        for (unsigned int s = 0; s < pomdp->n; s++) {
            hsvi2->lowerGamma[a * pomdp->n + s] = maxminR / (1.0f - pomdp->gamma);
        }
    }
    hsvi2->lowerGammaSize = pomdp->m;

    // Perform updates until convergence or a maximum number of iterations is reached equal to the horizon.
    // Note: Because this is essentially an MDP, we perform this operation on the same set of values.
    // It is essentially asynchronous value iteration, converging to a unique fixed point anyway.
    for (unsigned int a = 0; a < pomdp->m; a++) {
        float residual = hsvi2->epsilon + 1.0f;

        for (unsigned int i = 0; i < pomdp->horizon && residual >= hsvi2->epsilon; i++) {
            residual = 0.0f;

            for (unsigned int s = 0; s < pomdp->n; s++) {
                float value = 0.0f;

                for (unsigned int j = 0; j < pomdp->ns; j++) {
                    int sp = pomdp->S[s * pomdp->m * pomdp->ns + a * pomdp->ns + j];
                    if (sp < 0) {
                        break;
                    }
                    value += pomdp->T[s * pomdp->m * pomdp->ns + a * pomdp->ns + j] * hsvi2->lowerGamma[a * pomdp->n + sp];
                }

                value = pomdp->R[s * pomdp->m + a] + pomdp->gamma * value;

                residual = std::max(residual, std::fabs(hsvi2->lowerGamma[a * pomdp->n + s] - value));

                hsvi2->lowerGamma[a * pomdp->n + s] = value;
            }
        }
    }

    return NOVA_SUCCESS;
}


int pomdp_hsvi2_upper_bound_initialize_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2)
{
    if (hsvi2->maxAlphaVectors < pomdp->n) {
        fprintf(stderr, "Error[pomdp_hsvi2_upper_bound_initialize_cpu]: Cannot initialize upper bound. Not enough memory.");
        return NOVA_ERROR_OUT_OF_MEMORY;
    }

    //* Fast Max Version (My Idea) -- Honestly, I think this is faster without any downside.

    // Note: This has a slightly different, improved computation of the initial alpha^a_0. Instead of solving an MDP,
    // we just use the maximum possible value in that MDP. Then, we do the HSVI2 convergence. Since it is a fixed point,
    // and we just want the values to monotonically decrease to this point, we can use this faster initial value.

    // First, we compute the absolute maximum values possible V_max = max_a max_s R(s, a) / (1 - gamma).
    float maxV = NOVA_FLT_MIN;
    for (unsigned int a = 0; a < pomdp->m; a++) {
        for (unsigned int s = 0; s < pomdp->n; s++) {
            if (pomdp->R[s * pomdp->m + a] > maxV) {
                double value = pomdp->R[s * pomdp->m + a] / (1.0f - pomdp->gamma);
                if (value > maxV) {
                    maxV = value;
                }
            }
        }
    }

    // Assign alpha^a_0 = maxV to all initial alpha values.
    float *alpha = new float[pomdp->m * pomdp->n];
    float *alphaPrime = new float[pomdp->m * pomdp->n];

    for (unsigned int i = 0; i < pomdp->m * pomdp->n; i++) {
        alpha[i] = maxV;
    }
    //*/

    /* MDP Version -- This is from the original paper. It works, but is slower and has no benefit... I think.
    // Construct and solve the MDP with value iteration.
    MDP mdp;
    mdp.n = pomdp->n;
    mdp.ns = pomdp->ns;
    mdp.m = pomdp->m;
    mdp.gamma = pomdp->gamma;
    mdp.horizon = pomdp->horizon;
    mdp.epsilon = hsvi2->epsilon;
    mdp.s0 = 0;
    mdp.ng = 0;
    mdp.goals = nullptr;
    mdp.S = pomdp->S;
    mdp.T = pomdp->T;
    mdp.R = pomdp->R;
    mdp.d_S = nullptr;
    mdp.d_T = nullptr;
    mdp.d_R = nullptr;

    MDPVICPU mdpVI;
    mdpVI.VInitial = nullptr;
    mdpVI.currentHorizon = 0;
    mdpVI.V = nullptr;
    mdpVI.VPrime = nullptr;
    mdpVI.pi = nullptr;

    MDPValueFunction mdpPolicy;
    mdpPolicy.n = pomdp->n;
    mdpPolicy.m = pomdp->m;
    mdpPolicy.r = 0;
    mdpPolicy.S = nullptr;
    mdpPolicy.V = nullptr;
    mdpPolicy.pi = nullptr;

    int result = mdp_vi_execute_cpu(&mdp, &mdpVI, &mdpPolicy);

    // Assign alpha^a_0 = maxV to all initial alpha values.
    float *alpha = new float[pomdp->m * pomdp->n];
    float *alphaPrime = new float[pomdp->m * pomdp->n];

    for (unsigned int a = 0; a < pomdp->m; a++) {
        for (unsigned int s = 0; s < pomdp->n; s++) {
            alpha[a * pomdp->n + s] = mdpPolicy.V[s];
            alphaPrime[a * pomdp->n + s] = mdpPolicy.V[s];
        }
    }

    result = mdp_value_function_uninitialize(&mdpPolicy);
    //*/

    // Perform updates until convergence or a maximum number of iterations is reached equal to the horizon.
    float residual = hsvi2->epsilon + 1.0f;
    for (unsigned int i = 0; i < pomdp->horizon && residual >= hsvi2->epsilon; i++) {
        // Before each time step update, copy the current alpha values.
        for (unsigned int k = 0; k < pomdp->m * pomdp->n; k++) {
            alphaPrime[k] = alpha[k];
        }

        residual = 0.0f;

        for (unsigned int a = 0; a < pomdp->m; a++) {
            for (unsigned int s = 0; s < pomdp->n; s++) {
                float value = 0.0f;

                for (unsigned int o = 0; o < pomdp->z; o++) {
                    float maxTransitionAlpha = NOVA_FLT_MIN;

                    // Note: The fast informed bound (FIB) computes the max over the alpha-vectors, but here we have one
                    // alpha-vector for each action.
                    for (unsigned int k = 0; k < pomdp->m; k++) {
                        float valueTransitionAlpha = 0.0f;

                        for (unsigned int j = 0; j < pomdp->ns; j++) {
                            int sp = pomdp->S[s * pomdp->m * pomdp->ns + a * pomdp->ns + j];
                            if (sp < 0) {
                                break;
                            }
                            valueTransitionAlpha += pomdp->O[a * pomdp->n * pomdp->z + sp * pomdp->z + o] * pomdp->T[s * pomdp->m * pomdp->ns + a * pomdp->ns + j] * alphaPrime[k * pomdp->n + sp];
                        }

                        if (k == 0 || valueTransitionAlpha > maxTransitionAlpha) {
                            maxTransitionAlpha = valueTransitionAlpha;
                        }
                    }

                    value += maxTransitionAlpha;
                }

                value = pomdp->R[s * pomdp->m + a] + pomdp->gamma * value;

                residual = std::max(residual, std::fabs(alphaPrime[a * pomdp->n + s] - value));

                alpha[a * pomdp->n + s] = value;
            }
        }
    }

    // After convergence, we assign the point set to be collapsed beliefs at each state with value max_a alpha^a(s).
    for (unsigned int s = 0; s < pomdp->n; s++) {
        hsvi2->upperGammaHVb[s] = NOVA_FLT_MIN;

        for (unsigned int a = 0; a < pomdp->m; a++) {
            for (unsigned int sp = 0; sp < pomdp->n; sp++) {
                if (s == sp) {
                    hsvi2->upperGammaB[s * pomdp->n + sp] = 1.0f;
                } else {
                    hsvi2->upperGammaB[s * pomdp->n + sp] = 0.0f;
                }
            }

            if (a == 0 || alpha[a * pomdp->n + s] > hsvi2->upperGammaHVb[s]) {
                hsvi2->upperGammaHVb[s] = alpha[a * pomdp->n + s];
            }
        }
    }
    hsvi2->upperGammaSize = pomdp->n;

    delete [] alpha;
    delete [] alphaPrime;
    alpha = nullptr;
    alphaPrime = nullptr;

    return NOVA_SUCCESS;
}


float pomdp_hsvi2_upper_Vb_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2, float *b)
{
    /*
    // As preprocessing, find the nearest corner; the one with the highest weight in b.
    unsigned int sNearest = 0;
    float bestValue = 0.0f;
    for (unsigned int s = 0; s < pomdp->n; s++) {
        if (b[s] > bestValue) {
            sNearest = s;
            bestValue = b[s];
        }
    }

    // Compute this corner's distance to b.
    float sNearestBeliefDistance = 0.0f;
    for (unsigned int s = 0; s < pomdp->n; s++) {
        sNearestBeliefDistance += std::pow((float)(s == sNearest) - b[s], 2);
    }
    sNearestBeliefDistance = std::sqrt(sNearestBeliefDistance);
    */

    float x0 = 0.0f;
    for (unsigned int s = 0; s < pomdp->n; s++) {
        x0 += hsvi2->upperGammaHVb[s] * b[s];
    }

    // For each upper bound element in the point set, we compute the interpolated point from this to the corners.
    // Note: The first n values of upperGamma are essentially reserved (and ordered) for the values of states.
    float upperVb = NOVA_FLT_MAX;

    for (unsigned int i = 0; i < hsvi2->upperGammaSize; i++) {
        /*
        // Approximately project the belief to the convex hull formed by the interior belief point i.
        // First, compute the weight between belief b_sNearest (a corner) and b_i (in upperGammaB) based
        // on the Euclidean distances.
        float iBeliefDistance = 0.0f;
        for (unsigned int s = 0; s < pomdp->n; s++) {
            iBeliefDistance += std::pow(hsvi2->upperGammaB[i * pomdp->n + s] - b[s], 2);
        }
        iBeliefDistance = std::sqrt(iBeliefDistance);

        // Special: If the belief i, the belief corner at s, and the belief b, are all basically the same, just continue.
        if (iBeliefDistance + sNearestBeliefDistance < 0.001f) {
            continue;
        }

        float iWeight = 1.0f - iBeliefDistance / (iBeliefDistance + sNearestBeliefDistance);

        // Lastly, compute the weighted value between these two points and take the min over all the belief i's.
        // Note: Again, this is special because the first n values are essentially reserved (and ordered by state).
        float interpolatedVb = iWeight * hsvi2->upperGammaHVb[i] + (1.0f - iWeight) * hsvi2->upperGammaHVb[sNearest];
        if (interpolatedVb < upperVb) {
            upperVb = interpolatedVb;
        }
        */

        float iWeight = NOVA_FLT_MAX;
        for (unsigned int s = 0; s < pomdp->n; s++) {
            if (hsvi2->upperGammaB[i * pomdp->n + s] > 0.0f) {
                float distance = b[s] / hsvi2->upperGammaB[i * pomdp->n + s];
                if (distance < iWeight) {
                    iWeight = distance;
                }
            }
        }

        float wTb = 0.0f;
        for (unsigned int s = 0; s < pomdp->n; s++) {
            wTb += hsvi2->upperGammaHVb[s] * hsvi2->upperGammaB[i * pomdp->n + s];
        }

        float xi = x0 + iWeight * (hsvi2->upperGammaHVb[i] - wTb);
        if (xi < upperVb) {
            upperVb = xi;
        }
    }

    return upperVb;
}


float pomdp_hsvi2_compute_prob_observation_given_belief_action_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2, float *b, unsigned int a, unsigned int o)
{
    float probObservationGivenBeliefAction = 0.0f;

    for (unsigned int s = 0; s < pomdp->n; s++) {
        for (unsigned int i = 0; i < pomdp->ns; i++) {
            int sp = pomdp->S[s * pomdp->m * pomdp->ns + a * pomdp->ns + i];
            if (sp < 0) {
                break;
            }
            probObservationGivenBeliefAction += pomdp->O[a * pomdp->n * pomdp->z + sp * pomdp->z + o] * pomdp->T[s * pomdp->m * pomdp->ns + a * pomdp->ns + i] * b[s];
        }
    }

    return probObservationGivenBeliefAction;
}


float pomdp_hsvi2_upper_Qba_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2, float *b, unsigned int a)
{
    float Qba = 0.0f;

    // First, iterate over all the observations and compute the expected value following this action.
    float *bp = nullptr;

    for (unsigned int o = 0; o < pomdp->z; o++) {
        // Note: pomdp_belief_update_cpu allocates memory for bp.
        int result = pomdp_belief_update_cpu(pomdp, b, a, o, bp);
        if (result != NOVA_SUCCESS) {
            continue;
        }

        Qba += pomdp_hsvi2_compute_prob_observation_given_belief_action_cpu(pomdp, hsvi2, b, a, o) * pomdp_hsvi2_upper_Vb_cpu(pomdp, hsvi2, bp);

        delete [] bp;
        bp = nullptr;
    }

    // Don't forget to scale this summation by gamma...
    Qba *= pomdp->gamma;

    // Second, add the reward R(b, a) and return.
    for (unsigned int s = 0; s < pomdp->n; s++) {
        Qba += b[s] * pomdp->R[s * pomdp->m + a];
    }

    return Qba;
}


float pomdp_hsvi2_width_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2, float *b)
{
    // The primary challenge is computing the upper bound on V(b) from the point set representation.
    float upperVb = pomdp_hsvi2_upper_Vb_cpu(pomdp, hsvi2, b);
    float lowerVb = NOVA_FLT_MIN;

    // Compute the lower bound on V(b) and return the width.
    for (unsigned int i = 0; i < hsvi2->lowerGammaSize; i++) {
        float bDotAlpha = 0.0f;
        for (unsigned int s = 0; s < pomdp->n; s++) {
            bDotAlpha += b[s] * hsvi2->lowerGamma[i * pomdp->n + s];
        }

        if (lowerVb < bDotAlpha) {
            lowerVb = bDotAlpha;
        }
    }

    return upperVb - lowerVb;
}


unsigned int pomdp_hsvi2_compute_a_star_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2, float *b)
{
    // Compute the argmax over the actions for the upper bound's Q-values given this action at the belief given.
    unsigned int aStar = 0;
    float bestQba = NOVA_FLT_MIN;

    for (unsigned int a = 0; a < pomdp->m; a++) {
        float Qba = pomdp_hsvi2_upper_Qba_cpu(pomdp, hsvi2, b, a);
        if (Qba > bestQba) {
            bestQba = Qba;
            aStar = a;
        }
    }

    return aStar;
}


unsigned int pomdp_hsvi2_compute_o_star_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2, float *b, unsigned int aStar, float gammaPowerT)
{
    unsigned int oStar = 0;
    float bestValue = NOVA_FLT_MIN;

    float *bp = nullptr;

    for (unsigned int o = 0; o < pomdp->z; o++) {
        // Note: pomdp_belief_update_cpu allocates memory for bp.
        int result = pomdp_belief_update_cpu(pomdp, b, aStar, o, bp);
        if (result != NOVA_SUCCESS) {
            if (bp != nullptr) {
                delete [] bp;
                bp = nullptr;
            }
            continue;
        }

        float excess = pomdp_hsvi2_width_cpu(pomdp, hsvi2, bp) - hsvi2->epsilon / (gammaPowerT * pomdp->gamma);
        float value = pomdp_hsvi2_compute_prob_observation_given_belief_action_cpu(pomdp, hsvi2, b, aStar, o) * excess;
        if (value > bestValue) {
            bestValue = value;
            oStar = o;
        }

        delete [] bp;
        bp = nullptr;
    }

    return oStar;
}


int pomdp_hsvi2_lower_bound_update_step_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2, float *b)
{
    if (hsvi2->lowerGammaSize >= hsvi2->maxAlphaVectors - 1) {
        fprintf(stderr, "Error[pomdp_hsvi2_lower_bound_update_step_cpu]: %s\n",
                "Cannot add any more lower bound alpha-vectors. Out of memory.");
        return NOVA_ERROR_OUT_OF_MEMORY;
    }

    // To hack the bellman update function, we need to use this Z. Also (re)define other variables.
    int *Z = new int[pomdp->n];
    for (unsigned int s = 0; s < pomdp->n; s++) {
        Z[s] = s;
    }
    float *B = b;
    unsigned int r = 1;
    unsigned int rz = pomdp->n;
    unsigned int bIndex = 0;

    pomdp_bellman_update_cpu(pomdp->n, pomdp->ns, pomdp->m, pomdp->z, r, rz, pomdp->gamma,
                             pomdp->S, pomdp->T, pomdp->O, pomdp->R, Z, B,
                             hsvi2->lowerGamma, hsvi2->lowerGammaSize,
                             bIndex, &hsvi2->lowerGamma[hsvi2->lowerGammaSize * pomdp->n], &hsvi2->lowerPi[hsvi2->lowerGammaSize]);
    hsvi2->lowerGammaSize++;

    // Cleanup temporary variables. Note: We do not need to cleanup B...
    delete [] Z;
    Z = nullptr;
    B = nullptr;

    return NOVA_SUCCESS;
}


int pomdp_hsvi2_upper_bound_update_step_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2, float *b)
{
    if (hsvi2->upperGammaSize >= hsvi2->maxAlphaVectors - 1) {
        fprintf(stderr, "Error[pomdp_hsvi2_upper_bound_update_step_cpu]: %s\n",
                "Cannot add any more upper bound points to the point set. Out of memory.");
        return NOVA_ERROR_OUT_OF_MEMORY;
    }

    memcpy(&hsvi2->upperGammaB[hsvi2->upperGammaSize * pomdp->n], b, pomdp->n * sizeof(float));
    hsvi2->upperGammaHVb[hsvi2->upperGammaSize] = pomdp_hsvi2_upper_Vb_cpu(pomdp, hsvi2, b);
    hsvi2->upperGammaSize++;

    return NOVA_SUCCESS;
}


void pomdp_hsvi2_lower_bound_prune_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2)
{
    // We will not waste time doing pruning if: (1) there are no lower bound alpha-vectors, or (2) you have pruned before
    // but the size has not grown enough (i.e., greater than the growth threshold).
    if (hsvi2->lowerGammaSize == 0 || (hsvi2->lowerGammaSizeLastPruned > 0 &&
            (float)hsvi2->lowerGammaSize / (float)hsvi2->lowerGammaSizeLastPruned <= 1.0f + hsvi2->pruneGrowthThreshold)) {
        return;
    }

    // For all alpha-vectors, we check all other alpha-vectors to see if one point-wise dominates. If so, then we prune it.
    for (unsigned int i = 0; i < hsvi2->lowerGammaSize; i++) {
        bool pointWiseDominated = false;

        for (unsigned int j = 0; j < hsvi2->lowerGammaSize; j++) {
            if (i == j) {
                continue;
            }

            bool allStatesDominated = true;
            for (unsigned int s = 0; s < pomdp->n; s++) {
                if (hsvi2->lowerGamma[i * pomdp->n + s] > hsvi2->lowerGamma[j * pomdp->n + s]) {
                    allStatesDominated = false;
                    break;
                }
            }

            if (allStatesDominated) {
                pointWiseDominated = true;
                break;
            }
        }

        if (pointWiseDominated) {
            // We prune by copying the last element in the array into the location of the dominated alpha-vector.
            memcpy(&hsvi2->lowerGamma[i * pomdp->n], &hsvi2->lowerGamma[(hsvi2->lowerGammaSize - 1) * pomdp->n], pomdp->n * sizeof(float));
            hsvi2->lowerPi[i] = hsvi2->lowerPi[hsvi2->lowerGammaSize - 1];
            hsvi2->lowerGammaSize--;
        }
    }

    hsvi2->lowerGammaSizeLastPruned = hsvi2->lowerGammaSize;
}


void pomdp_hsvi2_upper_bound_prune_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2)
{
    // We will not waste time doing pruning if: (1) there are no upper bound points in the point set, or (2) you have not
    // pruned before but the size has not grown enough (i.e., greater than the growth threshold).
    if (hsvi2->upperGammaSize == 0 || (hsvi2->upperGammaSizeLastPruned > 0 &&
            (float)hsvi2->upperGammaSize / (float)hsvi2->upperGammaSizeLastPruned <= 1.0f + hsvi2->pruneGrowthThreshold)) {
        return;
    }

    return;

    unsigned int *pointsToRemove = new unsigned int[hsvi2->upperGammaSize];
    unsigned int numPointsToRemove = 0;

    // For all elements in the point set, we check if the value at that belief already has a lower value that can be computed
    // by other elements in the point set. Special: The first n-values are never removed; only upgraded with a better HVb
    // found via exploration in the algorithm. Since these are unique beliefs from each other, they will never be pruned.
    for (unsigned int i = 0; i < hsvi2->upperGammaSize; i++) {
        numPointsToRemove = 0;

        // For each of the other upper points, find ones that are basically the same. For any such point, we compute the
        // minimum over the HVb values to use instead of all of those points.
        for (unsigned int j = i + 1; j < hsvi2->upperGammaSize; j++) {
            bool sameBeliefPoint = true;
            for (unsigned int s = 0; s < pomdp->n; s++) {
                if (std::fabs(hsvi2->upperGammaB[i * pomdp->n + s] - hsvi2->upperGammaB[j * pomdp->n + s]) >= 0.0001f) {
                    sameBeliefPoint = false;
                    break;
                }
            }

            if (sameBeliefPoint) {
                hsvi2->upperGammaHVb[i] = std::min(hsvi2->upperGammaHVb[i], hsvi2->upperGammaHVb[j]);
                pointsToRemove[numPointsToRemove] = j;
                numPointsToRemove++;
            }
        }

        // Remove all the points we found that are essentially duplicates and/or have worse values.
        for (unsigned int j = 0; j < numPointsToRemove; j++) {
            if (pointsToRemove[j] < hsvi2->upperGammaSize - 1) {
                memcpy(&hsvi2->upperGammaB[pointsToRemove[j] * pomdp->n], &hsvi2->upperGammaB[(hsvi2->upperGammaSize - 1) * pomdp->n], pomdp->n * sizeof(float));
                hsvi2->upperGammaHVb[j] = hsvi2->upperGammaHVb[hsvi2->upperGammaSize - 1];
            }
            hsvi2->upperGammaSize--;
        }
    }

    delete [] pointsToRemove;
    pointsToRemove = nullptr;

    hsvi2->upperGammaSizeLastPruned = hsvi2->upperGammaSize;
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
    hsvi2->lowerGammaSize = 0;
    hsvi2->lowerGammaSizeLastPruned = 0;
    hsvi2->lowerGamma = new float[hsvi2->maxAlphaVectors * pomdp->n];
    hsvi2->lowerPi = new unsigned int[hsvi2->maxAlphaVectors];

    hsvi2->upperGammaSize = 0;
    hsvi2->upperGammaSizeLastPruned = 0;
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
    hsvi2->lowerGammaSizeLastPruned = 0;
    hsvi2->upperGammaSize = 0;
    hsvi2->upperGammaSizeLastPruned = 0;

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
    for (unsigned int s = 0; s < pomdp->n; s++) {
        b[s] = 0.0f;
    }
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
        unsigned int aStar = pomdp_hsvi2_compute_a_star_cpu(pomdp, hsvi2, b);

        // Select an observation according to the forward exploration heuristics.
        unsigned int oStar = pomdp_hsvi2_compute_o_star_cpu(pomdp, hsvi2, b, aStar, gammaPowerT);

        // Push the current belief point on the "traversal belief stack".
        memcpy(&traversalBeliefStack[currentHorizon * pomdp->n], b, pomdp->n * sizeof(float));

        // Transition to the new belief given this action-observation. Note: The update allocates more memory.
        delete [] b;
        b = nullptr;

        int updateResult = pomdp_belief_update_cpu(pomdp, &traversalBeliefStack[currentHorizon * pomdp->n], aStar, oStar, b);
        if (updateResult != NOVA_SUCCESS) {
            fprintf(stderr, "Warning[pomdp_hsvi2_update_cpu]: %s\n",
                    "Invalid action-observation pair chosen in HSVI2 exploration step. Terminating early.");
            break;
        }
    }

    // In post-order traversal, perform an update on each belief points for both the lower and upper bounds.
    bool ranOutOfMemory = false;

    for (int i = std::max((int)currentHorizon - 1, 0); i >= 0 && !ranOutOfMemory; i--) {
        int lowerResult = pomdp_hsvi2_lower_bound_update_step_cpu(pomdp, hsvi2, &traversalBeliefStack[i * pomdp->n]);
        int upperResult = pomdp_hsvi2_upper_bound_update_step_cpu(pomdp, hsvi2, &traversalBeliefStack[i * pomdp->n]);

        if (lowerResult == NOVA_ERROR_OUT_OF_MEMORY || upperResult == NOVA_ERROR_OUT_OF_MEMORY) {
            ranOutOfMemory = true;
            break;
        }

        // Also, prune extra lower bound alpha-vectors and upper bound points. Note: These functions do the check as well.
        pomdp_hsvi2_lower_bound_prune_cpu(pomdp, hsvi2);
        pomdp_hsvi2_upper_bound_prune_cpu(pomdp, hsvi2);
    }

    // The entire algorithm has converged if the width(b0) is ever less than epsilon.
    bool converged = false;
    float width = pomdp_hsvi2_width_cpu(pomdp, hsvi2, &traversalBeliefStack[0]);
    if (width < hsvi2->epsilon) {
        converged = true;
    }

    // Cleanup memory and return the appropriate error (e.g., converged, ran out of memory, or just success).
    delete [] traversalBeliefStack;
    delete [] b;
    traversalBeliefStack = nullptr;
    b = nullptr;

    if (converged) {
        return NOVA_CONVERGED;
    } else if (ranOutOfMemory) {
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



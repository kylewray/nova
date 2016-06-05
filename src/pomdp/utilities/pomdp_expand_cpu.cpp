/**
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2015 Kyle Hollins Wray, University of Massachusetts
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


#include <nova/pomdp/utilities/pomdp_expand_cpu.h>

#include <stdio.h>
#include <cstring>
#include <cstdlib>
#include <time.h>
#include <cmath>

#include <nova/pomdp/utilities/pomdp_model_cpu.h>
#include <nova/error_codes.h>
#include <nova/constants.h>

namespace nova {

int pomdp_expand_construct_belief_cpu(const POMDP *pomdp, unsigned int i, float *b)
{
    for (unsigned int s = 0; s < pomdp->n; s++) {
        b[s] = 0.0f;
    }

    for (unsigned int j = 0; j < pomdp->rz; j++) {
        int s = pomdp->Z[i * pomdp->rz + j];
        if (s < 0) {
            break;
        }
        b[s] = pomdp->B[i * pomdp->rz + j];
    }

    return NOVA_SUCCESS;
}


int pomdp_expand_probability_observation_cpu(const POMDP *pomdp, const float *b, unsigned int a,
    unsigned int o, float &prObs)
{
    prObs = 0.0f;

    for (unsigned int s = 0; s < pomdp->n; s++) {
        float val = 0.0f;

        for (unsigned int i = 0; i < pomdp->ns; i++) {
            int sp = pomdp->S[s * pomdp->m * pomdp->ns + a * pomdp->ns + i];
            if (sp < 0) {
                break;
            }

            val += pomdp->T[s * pomdp->m * pomdp->ns + a * pomdp->ns + i] *
                   pomdp->O[a * pomdp->n * pomdp->z + sp * pomdp->z + o];
        }

        prObs += val * b[s];
    }

    return NOVA_SUCCESS;
}


int pomdp_expand_random_cpu(POMDP *pomdp, unsigned int numBeliefsToAdd)
{
    // Ensure the data is valid.
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            pomdp->Z == nullptr || pomdp->B == nullptr || pomdp->horizon < 1 ||
            numBeliefsToAdd == 0) {
        fprintf(stderr, "Error[pomdp_expand_random_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    srand(time(nullptr));

    // This is our belief used in each step of the for loop.
    float *b = new float[pomdp->n];
    unsigned int i = 0;

    // Create the new beliefs matrix.
    float *Bnew = new float[numBeliefsToAdd * pomdp->n];
    for (unsigned int i = 0; i < numBeliefsToAdd * pomdp->n; i++) {
        Bnew[i] = 0.0f;
    }

    // For each belief point we want to expand. Each step will generate a new trajectory
    // and add the resulting belief point to B.
    while (i < numBeliefsToAdd) {
        // Setup the belief used in exploration. We select a random belief as our initial belief.
        unsigned int b0Index = (unsigned int)((float)rand() / (float)RAND_MAX * (float)pomdp->r);
        pomdp_expand_construct_belief_cpu(pomdp, b0Index, b);

        // Randomly pick a horizon for this trajectory. We do this because some domains transition
        // beliefs away from areas on the (n-1)-simplex, never to return. This ensures many paths
        // are added.
        unsigned int h = (unsigned int)((float)rand() / (float)RAND_MAX * (float)(pomdp->horizon + 1));

        // Follow a random trajectory with length equal to this horizon.
        for (unsigned int t = 0; t < h; t++) {
            // Randomly pick an action.
            unsigned int a = (unsigned int)((float)rand() / (float)RAND_MAX * (float)pomdp->m);

            float currentNumber = 0.0f;
            float targetNumber = (float)rand() / (float)RAND_MAX;

            unsigned int o = 0;
            for (unsigned int op = 0; op < pomdp->z; op++) {
                float prObs = 0.0f;
                pomdp_expand_probability_observation_cpu(pomdp, b, a, op, prObs);
                currentNumber += prObs;

                // Note: We ensure that whatever observation is observed can actually happen.
                if (currentNumber >= targetNumber && prObs > 0.0f) {
                    o = op;
                    break;
                }
            }

            // Follow the belief update equation to compute b' for all state primes s'.
            float *bp = nullptr;
            pomdp_belief_update_cpu(pomdp, b, a, o, bp);
            memcpy(b, bp, pomdp->n * sizeof(float));
            delete [] bp;

            // Assign the computed belief for this trajectory.
            memcpy(&Bnew[i * pomdp->n], b, pomdp->n * sizeof(float));

            // Stop if we have met the belief point quota.
            i++;
            if (i >= numBeliefsToAdd) {
                break;
            }
        }
    }

    // Finally, we update r, rz, and B with Bnew.
    pomdp_add_new_raw_beliefs_cpu(pomdp, numBeliefsToAdd, Bnew);

    delete [] b;
    delete [] Bnew;

    return NOVA_SUCCESS;
}


int pomdp_expand_distinct_beliefs_cpu(POMDP *pomdp)
{
    // Ensure the data is valid.
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            pomdp->Z == nullptr || pomdp->B == nullptr) {
        fprintf(stderr, "Error[pomdp_expand_distinct_beliefs_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // The number of beliefs to add.
    unsigned int numBeliefsToAdd = pomdp->r;

    // Create the new beliefs matrix.
    float *Bnew = new float[numBeliefsToAdd * pomdp->n];
    for (unsigned int i = 0; i < numBeliefsToAdd * pomdp->n; i++) {
        Bnew[i] = 0.0f;
    }

    float *b = new float[pomdp->n];
    for (unsigned int i = 0; i < pomdp->n; i++) {
        b[i] = 0.0f;
    }

    for (unsigned int i = 0; i < numBeliefsToAdd; i++) {
        // Construct a belief to use in computing b'.
        pomdp_expand_construct_belief_cpu(pomdp, i, b);

        float *bpStar = new float[pomdp->n];
        float bpStarValue = FLT_MIN;

        // For this belief point, find the action-observation pair that is most distinct
        // from the current set of beliefs B. This will be added to Bnew.
        for (unsigned int a = 0; a < pomdp->m; a++) {
            for (unsigned int o = 0; o < pomdp->z; o++) {
                // Ensure that this observation is possible given the belief and action.
                float prObs = 0.0f;
                pomdp_expand_probability_observation_cpu(pomdp, b, a, o, prObs);
                if (prObs == 0.0f) {
                    continue;
                }

                // Compute b'.
                float *bp = nullptr;
                unsigned int result = pomdp_belief_update_cpu(pomdp, b, a, o, bp);

                // Since we are iterating over observations, we may examine an observation
                // which is impossible given the current belief. This is based on the POMDP.
                // Thus, this is an invalid successor belief state, so continue. Technically,
                // this should be caught by the first check above, but this is just in case.
                if (result != NOVA_SUCCESS) {
                    if (bp != nullptr) {
                        delete [] bp;
                    }
                    continue;
                }

                // Compute min|b - b'| for all beliefs b in B.
                float jStarValue = FLT_MAX;

                for (unsigned int j = 0; j < pomdp->r; j++) {
                    float *btmp = new float[pomdp->n];
                    pomdp_expand_construct_belief_cpu(pomdp, j, btmp);

                    float jValue = 0.0f;
                    for (unsigned int s = 0; s < pomdp->n; s++) {
                        jValue += std::fabs(btmp[s] - bp[s]);
                    }

                    if (jValue < jStarValue) {
                        jStarValue = jValue;
                    }

                    delete [] btmp;
                }

                // Now, determine if this was the largest b' we found so far. If so, remember it.
                if (jStarValue > bpStarValue) {
                    memcpy(bpStar, bp, pomdp->n * sizeof(float));
                    bpStarValue = jStarValue;
                }

                delete [] bp;
            }
        }

        // For this belief b, add a new belief bp* = max_{a, o} min_{b'' in B} |b'(b, a, o) - b''|_1.
        memcpy(&Bnew[i * pomdp->n], bpStar, pomdp->n * sizeof(float));

        delete [] bpStar;
    }

    // Finally, we update r, rz, and B with Bnew.
    pomdp_add_new_raw_beliefs_cpu(pomdp, numBeliefsToAdd, Bnew);

    delete [] b;
    delete [] Bnew;

    return NOVA_SUCCESS;
}


int pomdp_expand_pema_cpu(POMDP *pomdp, const POMDPAlphaVectors *policy)
{
    // Ensure the data is valid.
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            pomdp->Z == nullptr || pomdp->B == nullptr) {
        fprintf(stderr, "Error[pomdp_expand_pema_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // The number of beliefs to add.
    unsigned int numBeliefsToAdd = 1;

    // Create the new beliefs matrix.
    float *Bnew = new float[numBeliefsToAdd * pomdp->n];
    for (unsigned int i = 0; i < numBeliefsToAdd * pomdp->n; i++) {
        Bnew[i] = 0.0f;
    }

    float bStarEpsilonErrorBound = FLT_MIN;

    // Compute Rmin and Rmax.
    float Rmax = FLT_MIN;
    float Rmin = FLT_MAX;
    for (unsigned int s = 0; s < pomdp->n; s++) {
        for (unsigned int a = 0; a < pomdp->m; a++) {
            if (Rmax < pomdp->R[s * pomdp->m + a]) {
                Rmax = pomdp->R[s * pomdp->m + a];
            }
            if (Rmin > pomdp->R[s * pomdp->m + a]) {
                Rmin = pomdp->R[s * pomdp->m + a];
            }
        }
    }

    float *b = new float[pomdp->n];
    unsigned int *oStar = new unsigned int[pomdp->m];
    float *oStarValue = new float[pomdp->m];

    for (unsigned int i = 0; i < pomdp->r; i++) {
        unsigned int aStar = 0;
        float aStarValue = FLT_MIN;

        pomdp_expand_construct_belief_cpu(pomdp, i, b);

        // During computing the max action, we will store the observation which introduced
        // the maximal error, i.e., one with highest value of epsilon(b'(b, a, o)).
        for (unsigned int a = 0; a < pomdp->m; a++) {
            oStar[a] = 0;
            oStarValue[a] = FLT_MIN;
        }

        // At this belief b, select the action which maximizes the sum over observations
        // of the probability of this observation times the expected worst-case error
        // at the *next* belief b' following a and o.
        for (unsigned int a = 0; a < pomdp->m; a++) {
            float aValue = 0.0f;

            for (unsigned int o = 0; o < pomdp->z; o++) {
                // Compute Pr(o|b,a).
                float probObsGivenBeliefAction = 0.0f;

                for (unsigned int j = 0; j < pomdp->rz; j++) {
                    int s = pomdp->Z[i * pomdp->rz + j];
                    if (s < 0) {
                        break;
                    }

                    for (unsigned int k = 0; k < pomdp->ns; k++) {
                        int sp = pomdp->S[s * pomdp->m * pomdp->ns + a * pomdp->ns + k];
                        if (sp < 0) {
                            break;
                        }

                        probObsGivenBeliefAction += pomdp->T[s * pomdp->m * pomdp->ns + a * pomdp->ns + k] *
                                                    pomdp->O[a * pomdp->n * pomdp->z + sp * pomdp->z + o] *
                                                    pomdp->B[i * pomdp->rz + j];
                    }
                }

                // With the current action and observation compute b'(b, a, o).
                float *bp = nullptr;
                unsigned int result = pomdp_belief_update_cpu(pomdp, b, a, o, bp);

                // Since we are iterating over observations, we may examine an observation
                // which is impossible given the current belief. This is based on the POMDP.
                // Thus, this is an invalid successor belief state, so continue.
                if (result != NOVA_SUCCESS) {
                    if (bp != nullptr) {
                        delete [] bp;
                    }
                    continue;
                }

                // Compute the closest (1-norm) belief from b'.
                unsigned int bClosestIndex = 0;
                float *bClosest = new float[pomdp->n];
                float bClosestStarDistance = FLT_MAX;

                for (unsigned int j = 0; j < pomdp->r; j++) {
                    float *bCheck = new float[pomdp->n];
                    pomdp_expand_construct_belief_cpu(pomdp, j, bCheck);

                    float bCheckDistance = 0.0f;
                    for (unsigned int s = 0; s < pomdp->n; s++) {
                        bCheckDistance += std::fabs(bp[s] - bCheck[s]);
                    }

                    if (bCheckDistance < bClosestStarDistance) {
                        bClosestIndex = j;
                        memcpy(bClosest, bCheck, pomdp->n * sizeof(float));
                        bClosestStarDistance = bCheckDistance;
                    }

                    delete [] bCheck;
                }

                // Compute alpha = argmax_{alpha in Gamma} alpha * b.
                float VbStar = FLT_MIN;
                unsigned int alphaIndexStar = 0;

                for (unsigned int j = 0; j < policy->r; j++) {
                    float Vb = 0.0f;

                    for (unsigned int s = 0; s < policy->n; s++) {
                        Vb = policy->Gamma[j * policy->n + s] *  bClosest[s];

                        if (Vb < VbStar) {
                            Vb = VbStar;
                            alphaIndexStar = j;
                        }
                    }
                }

                // Compute epsilon(b'(bClosest, a, o)).
                float epsilonBeliefPrime = 0.0f;
                for (unsigned int s = 0; s < pomdp->n; s++) {
                    if (bp[s] >= b[s]) {
                        epsilonBeliefPrime += (Rmax / (1.0f - pomdp->gamma) -
                                policy->Gamma[alphaIndexStar * pomdp->n + s]) * (bp[s] - bClosest[s]);
                    } else {
                        epsilonBeliefPrime += (Rmin / (1.0f - pomdp->gamma) -
                                policy->Gamma[alphaIndexStar * pomdp->n + s]) * (bp[s] - bClosest[s]);
                    }
                }

                // For this action, update the maximal oStar[a].
                if (probObsGivenBeliefAction * epsilonBeliefPrime > oStarValue[a]) {
                    oStar[a] = o;
                    oStarValue[a] = probObsGivenBeliefAction * epsilonBeliefPrime;
                }

                // Add the result to find the maximal action.
                aValue += probObsGivenBeliefAction * epsilonBeliefPrime;

                delete [] bp;
                delete [] bClosest;
            }

            if (aValue > aStarValue) {
                aStar = a;
                aStarValue = aValue;
            }
        }

        // We are looking for the largest value possible.
        if (aStarValue > bStarEpsilonErrorBound) {
            // With the best action, compute b'(b, aStar, oStar[aStar]) from the initial b (from i).
            // In the rare case of a failure, just copy the original belief.
            float *bp = nullptr;
            unsigned int result = pomdp_belief_update_cpu(pomdp, b, aStar, oStar[aStar], bp);

            if (result == NOVA_SUCCESS) {
                memcpy(&Bnew[0 * pomdp->n], bp, pomdp->n * sizeof(float));
            } else {
                memcpy(&Bnew[0 * pomdp->n], b, pomdp->n * sizeof(float));
            }

            if (bp != nullptr) {
                delete [] bp;
            }

            bStarEpsilonErrorBound = aStarValue;
        }
    }

    delete [] b;
    delete [] oStar;
    delete [] oStarValue;

    // Finally, we update r, rz, and B with Bnew.
    pomdp_add_new_raw_beliefs_cpu(pomdp, numBeliefsToAdd, Bnew);

    delete [] Bnew;

    return NOVA_SUCCESS;
}

}; // namespace nova


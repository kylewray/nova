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


#include "utilities/pomdp_expand_gpu.h"
#include "error_codes.h"
#include "constants.h"

#include <stdio.h>
#include <cstring>
#include <cstdlib>
#include <time.h>
#include <cmath>


int pomdp_expand_construct_belief_gpu(const POMDP *pomdp, unsigned int i, float *b)
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


int pomdp_expand_belief_update_gpu(const POMDP *pomdp, const float *b, unsigned int a,
    unsigned int o, float *bp)
{
    for (unsigned int sp = 0; sp < pomdp->n; sp++) {
        bp[sp] = 0.0f;
    }

    for (unsigned int s = 0; s < pomdp->n; s++) {
        for (unsigned int i = 0; i < pomdp->ns; i++) {
            int sp = pomdp->S[s * pomdp->m * pomdp->ns + a * pomdp->ns + i];
            if (sp < 0) {
                break;
            }

            bp[sp] += pomdp->T[s * pomdp->m * pomdp->ns + a * pomdp->ns + i] * b[s];
        }
    }

    float normalizingConstant = 0.0f;

    for (unsigned int sp = 0; sp < pomdp->n; sp++) {
        bp[sp] *= pomdp->O[a * pomdp->n * pomdp->z + sp * pomdp->z + o];
        normalizingConstant += bp[sp];
    }

    // If the normalizing constant is exceedingly small, within error tolerances, then this is
    // very likely to be an invalid belief. In practice, this arises when there is a probabilistically
    // impossible observation, given the POMDP.
    if (std::fabs(normalizingConstant) < FLT_ERR_TOL) {
        return NOVA_WARNING_INVALID_BELIEF;
    }

    for (unsigned int sp = 0; sp < pomdp->n; sp++) {
        bp[sp] /= normalizingConstant;
    }

    return NOVA_SUCCESS;
}


int pomdp_expand_probability_observation_gpu(const POMDP *pomdp, const float *b,
    unsigned int a, unsigned int o, float &prObs)
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


int pomdp_expand_update_max_non_zero_values_gpu(const POMDP *pomdp, const float *b,
    unsigned int *maxNonZeroValues)
{
    unsigned int numNonZeroValues = 0;
    for (unsigned int s = 0; s < pomdp->n; s++) {
        if (b[s] > 0.0f) {
            numNonZeroValues++;
        }
    }
    if (numNonZeroValues > *maxNonZeroValues) {
        *maxNonZeroValues = numNonZeroValues;
    }

    return NOVA_SUCCESS;
}


int pomdp_expand_random_gpu(const POMDP *pomdp, unsigned int numThreads,
    unsigned int numDesiredBeliefPoints,
    unsigned int *maxNonZeroValues, float *Bnew)
{
    srand(time(nullptr));

    *maxNonZeroValues = 0;

    // Setup the initial belief point.
    float *b0 = new float[pomdp->n];
    pomdp_expand_construct_belief_gpu(pomdp, 0, b0);

    float *b = new float[pomdp->n];
    unsigned int i = 1;

    // The first one is always the initial seed belief.
    memcpy(&Bnew[0 * pomdp->n], b0, pomdp->n * sizeof(float));

    // For each belief point we want to expand. Each step will generate a new trajectory
    // and add the resulting belief point to B.
    while (i < numDesiredBeliefPoints) {
        // Randomly pick a horizon for this trajectory. We do this because some domains transition
        // beliefs away from areas on the (n-1)-simplex, never to return. This ensures many paths
        // are added.
        unsigned int h = (unsigned int)((float)rand() / (float)RAND_MAX * (float)(pomdp->horizon + 1));

        // Setup the belief used in exploration.
        memcpy(b, b0, pomdp->n * sizeof(float));

        // Follow a random trajectory with length equal to this horizon.
        for (unsigned int t = 0; t < h; t++) {
            // Randomly pick an action.
            unsigned int a = (unsigned int)((float)rand() / (float)RAND_MAX * (float)pomdp->m);

            float currentNumber = 0.0f;
            float targetNumber = (float)rand() / (float)RAND_MAX;

            unsigned int o = 0;
            for (unsigned int op = 0; op < pomdp->z; op++) {
                float prObs = 0.0f;
                pomdp_expand_probability_observation_gpu(pomdp, b, a, op, prObs);
                currentNumber += prObs;

                if (currentNumber >= targetNumber) {
                    o = op;
                    break;
                }
            }

            // Follow the belief update equation to compute b' for all state primes s'.
            float *bp = new float[pomdp->n];
            pomdp_expand_belief_update_gpu(pomdp, b, a, o, bp);
            memcpy(b, bp, pomdp->n * sizeof(float));
            delete [] bp;

            // Determine how many non-zero values exist and update rz.
            pomdp_expand_update_max_non_zero_values_gpu(pomdp, b, maxNonZeroValues);

            // Assign the computed belief for this trajectory.
            memcpy(&Bnew[i * pomdp->n], b, pomdp->n * sizeof(float));

            // Stop if we have met the belief point quota.
            i++;
            if (i >= numDesiredBeliefPoints) {
                break;
            }
        }
    }

    delete [] b;
    delete [] b0;

    return NOVA_SUCCESS;
}



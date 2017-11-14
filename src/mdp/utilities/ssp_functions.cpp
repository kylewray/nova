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


#include <nova/mdp/utilities/ssp_functions.h>

#include <nova/mdp/mdp.h>
#include <nova/error_codes.h>
#include <nova/constants.h>

#include <stdio.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <math.h>

#include <iostream>

namespace nova {

void ssp_bellman_update(unsigned int n, unsigned int ns, unsigned int m, 
    const int *S, const float *T, const float *R, unsigned int s,
    float *V, unsigned int *pi)
{
    float VsPrime = NOVA_FLT_MAX;

    // Compute min_{a in A} Q(s, a). Recall, we are dealing with rewards R as positive costs.
    for (unsigned int a = 0; a < m; a++) {
        // Compute Q(s, a) for this action.
        float Qsa = R[s * m + a];

        for (unsigned int i = 0; i < ns; i++) {
            int sp = S[s * m * ns + a * ns + i];
            if (sp < 0) {
                break;
            }

            // Note: V is marked with a negative if it is solved, visited, etc., depending on the algorithm.
            Qsa += T[s * m * ns + a * ns + i] * std::fabs(V[sp]);
        }

        if (a == 0 || Qsa < VsPrime) {
            VsPrime = Qsa;
            pi[s] = a;
        }
    }

    V[s] = VsPrime;
}


int ssp_random_successor(const MDP *mdp, unsigned int s, unsigned int a, unsigned int &sp)
{
    if (mdp == nullptr || mdp->n == 0 || mdp->ns == 0 || mdp->m == 0 ||
            mdp->S == nullptr || mdp->T == nullptr) {
        fprintf(stderr, "Error[ssp_random_successor]: %s\n", "Invalid data.");
        return NOVA_ERROR_INVALID_DATA;
    }

    sp = mdp->S[s * mdp->m * mdp->ns + a * mdp->ns + 0];

    double target = (double)rand() / (double)RAND_MAX;
    double current = 0.0f;

    for (unsigned int i = 0; i < mdp->ns; i++) {
        int spTmp = mdp->S[s * mdp->m * mdp->ns + a * mdp->ns + i];
        if (spTmp < 0) {
            break;
        }

        current += mdp->T[s * mdp->m * mdp->ns + a * mdp->ns + i];

        if (current >= target) {
            sp = spTmp;
            break;
        }
    }

    return NOVA_SUCCESS;
}


bool ssp_is_goal(const MDP *mdp, unsigned int s)
{
    for (unsigned int i = 0; i < mdp->ng; i++) {
        if (s == mdp->goals[i]) {
            return true;
        }
    }

    return false;
}


bool ssp_is_dead_end(const MDP *mdp, unsigned int s)
{
    for (unsigned int a = 0; a < mdp->m; a++) {
        unsigned int sp = mdp->S[s * mdp->m * mdp->ns + a * mdp->ns + 0];
        float transitionProbability = mdp->T[s * mdp->m * mdp->ns + a * mdp->ns + 0];
        float cost = mdp->R[s * mdp->m + a];

        if (!(s == sp && transitionProbability == 1.0f && cost > 0.0f)) {
            return false;
        }
    }

    return true;
}

}; // namespace nova


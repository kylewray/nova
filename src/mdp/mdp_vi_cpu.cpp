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


#include "mdp_vi_cpu.h"
#include "error_codes.h"

#include <stdio.h>
#include <cstring>


// This is determined by hardware, so what is below is a 'safe' guess. If this is
// off, the program might return 'nan' or 'inf'.
#define FLT_MAX 1e+35


void mdp_bellman_update_cpu(unsigned int n, unsigned int ns, unsigned int m, float gamma, 
                        const int *S, const float *T, const float *R, const float *V,
                        float *VPrime, unsigned int *pi)
{
    // The intermediate Q(s, a) value.
    float Qsa;

    // The index within S and T (i.e., in n*s*ns).
    int index;

    // The true successor state index (in 0 to n-1), resolved using S.
    int spindex;

    for (unsigned int s = 0; s < n; s++) {
        VPrime[s] = -FLT_MAX;

        // Compute max_{a in A} Q(s, a).
        for (int a = 0; a < m; a++) {
            // Compute Q(s, a) for this action.
            Qsa = R[s * m + a];

            for (int sp = 0; sp < ns; sp++) {
                index = s * m * ns + a * ns + sp;

                spindex = S[index];
                if (spindex < 0) {
                    break;
                }

                Qsa += gamma * T[index] * V[spindex];
            }

            if (a == 0 || Qsa > VPrime[s]) {
                VPrime[s] = Qsa;
                pi[s] = a;
            }
        }
    }
}


int mdp_vi_complete_cpu(MDP *mdp, float *V, unsigned int *pi)
{
    float *VPrime;

    // First, ensure data is valid.
    if (mdp->n == 0 || mdp->ns == 0 || mdp->m == 0 ||
            mdp->S == nullptr || mdp->T == nullptr || mdp->R == nullptr ||
            mdp->gamma < 0.0f || mdp->gamma >= 1.0f || mdp->horizon < 1 ||
            V == nullptr || pi == nullptr) {
        return NOVA_ERROR_INVALID_DATA;
    }

    VPrime = new float[mdp->n];

    // We iterate over all time steps up to the horizon.
    for (int i = 0; i < mdp->horizon; i++) {
        printf("Iteration %d / %d -- CPU Version\n", i, mdp->horizon);

        // We oscillate between V and VPrime depending on the step.
        if (i % 2 == 0) {
            mdp_bellman_update_cpu(mdp->n, mdp->ns, mdp->m, mdp->gamma, mdp->S, mdp->T, mdp->R, V, VPrime, pi);
        } else {
            mdp_bellman_update_cpu(mdp->n, mdp->ns, mdp->m, mdp->gamma, mdp->S, mdp->T, mdp->R, VPrime, V, pi);
        }
    }

    // If the horizon was odd, then we must copy the value back to V from VPrime.
    // Otherwise, it was already stored in V.
    if (mdp->horizon % 2 == 1) {
        memcpy(V, VPrime, mdp->n * sizeof(float));
    }

    delete [] VPrime;

    return NOVA_SUCCESS;
}


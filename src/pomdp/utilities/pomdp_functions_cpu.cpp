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


#include <nova/pomdp/utilities/pomdp_functions_cpu.h>

#include <stdio.h>
#include <cstring>
#include <cstdlib>
#include <time.h>

#include <nova/error_codes.h>
#include <nova/constants.h>

namespace nova {

float pomdp_compute_b_dot_alpha_cpu(unsigned int rz, const int *Z, const float *B, unsigned int bIndex, const float *alpha)
{
    float bDotAlpha = 0.0f;

    for (unsigned int j = 0; j < rz; j++) {
        int s = Z[bIndex * rz + j];
        if (s < 0) {
            break;
        }

        bDotAlpha += B[bIndex * rz + j] * alpha[s];
    }

    return bDotAlpha;
}


void pomdp_compute_Vb_cpu(unsigned int n, unsigned int rz, const int *Z, const float *B, unsigned int bIndex,
    const float *Gamma, unsigned int rGamma, float *Vnb, unsigned int *alphaPrimeIndex)
{
    *Vnb = NOVA_FLT_MIN;

    for (unsigned int i = 0; i < rGamma; i++) {
        float bDotAlpha = pomdp_compute_b_dot_alpha_cpu(rz, Z, B, bIndex, &Gamma[i * n]);

        if (*Vnb < bDotAlpha) {
            *Vnb = bDotAlpha;
            *alphaPrimeIndex = i;
        }
    }
}


void pomdp_update_compute_best_alpha_cpu(unsigned int n, unsigned int ns, unsigned int m, unsigned int z,
    unsigned int r, unsigned int rz, float gamma,
    const int *S, const float *T, const float *O, const float *R, const int *Z, const float *B,
    unsigned int bIndex, const float *Gamma, unsigned int rGamma, unsigned int a, float *alpha)
{
    for (unsigned int o = 0; o < z; o++) {
        float value = NOVA_FLT_MIN;
        float bestValue = NOVA_FLT_MIN;
        unsigned int bestAlphaIndex = 0;

        for (unsigned int j = 0; j < rGamma; j++) {
            // Variable 'j' represents the alpha in Gamma^{t-1}. It is this variable that we will maximize over.

            // Compute the value of this alpha-vector, by taking its dot product with belief (i.e., variable 'i').
            value = 0.0f;
            for (unsigned int k = 0; k < rz; k++) {
                int s = Z[bIndex * rz + k];
                if (s < 0) {
                    break;
                }

                // Compute the updated value for this alpha vector from the previous time step, i.e.,
                // V^{t-1}(s, a, omega (o), alpha (j)) for all states s in S. Only do this for the non-zero
                // values, since we are just computing the value now.
                float Vtk = 0.0f;
                for (unsigned int l = 0; l < ns; l++) {
                    int sp = S[s * m * ns + a * ns + l];
                    if (sp < 0) {
                        break;
                    }

                    Vtk += O[a * n * z + sp * z + o] * T[s * m * ns + a * ns + l] * Gamma[j * n + sp];
                }
                Vtk *= gamma;

                value += Vtk * B[bIndex * rz + k];
            }

            // Assign this as the best alpha-vector if it is better.
            if (value > bestValue) {
                bestAlphaIndex = j;
                bestValue = value;
            }
        }

        // Now that we have the best 'j' (alpha-vector), we can compute the value over *all* states and add
        // this to the current 'alpha' variable we are computing. (This step is the final step of summing
        // over beliefs of the argmax of Vt.)
        for (unsigned int s = 0; s < n; s++) {
            float Vtk = 0.0f;
            for (unsigned int l = 0; l < ns; l++) {
                int sp = S[s * m * ns + a * ns + l];
                if (sp < 0) {
                    break;
                }

                Vtk += O[a * n * z + sp * z + o] * T[s * m * ns + a * ns + l] * Gamma[bestAlphaIndex * n + sp];
            }
            Vtk *= gamma;

            alpha[s] += Vtk;
        }
    }
}


void pomdp_bellman_update_cpu(unsigned int n, unsigned int ns, unsigned int m, unsigned int z,
    unsigned int r, unsigned int rz, float gamma,
    const int *S, const float *T, const float *O, const float *R, const int *Z, const float *B,
    const float *Gamma, unsigned int rGamma,
    unsigned int bIndex, float *alphaPrime, unsigned int *aPrime)
{
    float value = NOVA_FLT_MIN;
    float bestValue = NOVA_FLT_MIN;

    float *alpha = new float[n];

    // Compute the argmax alpha-vector over Gamma_B. Since Gamma_B is created from the
    // m actions, we iterate over a in A.
    for (unsigned int a = 0; a < m; a++) {
        // Variable 'a' also refers to the alpha-vector being considered in the argmax for
        // the 'best' alpha-vector, as well as the action.

        // We create alpha which initially is the reward, and we will add the optimal
        // alpha-vector for each observation in the function below.
        for (unsigned int s = 0; s < n; s++) {
            alpha[s] = R[s * m + a];
        }

        // First, compute the argmax_{alpha in Gamma_{a,omega}} for each observation.
        pomdp_update_compute_best_alpha_cpu(n, ns, m, z, r, rz, gamma, S, T, O, R, Z, B,
                                            bIndex, Gamma, rGamma, a, alpha);

        // After the alpha-vector is computed, we must compute its value.
        value = pomdp_compute_b_dot_alpha_cpu(rz, Z, B, bIndex, alpha);

        // If this is a new best value, then store the alpha-vector.
        if (value > bestValue) {
            memcpy(alphaPrime, alpha, n * sizeof(float));
            *aPrime = a;
            bestValue = value;
        }
    }

    delete [] alpha;
    alpha = nullptr;
}

}; // namespace nova


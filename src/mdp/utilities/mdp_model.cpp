/**
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2016 Kyle Hollins Wray, University of Massachusetts
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


#include <nova/mdp/utilities/mdp_model.h>

#include <stdio.h>
#include <cstring>

#include <nova/error_codes.h>
#include <nova/constants.h>

namespace nova {

int mdp_initialize(MDP *mdp, unsigned int n, unsigned int ns, unsigned int m, float gamma,
    unsigned int horizon, float epsilon, unsigned int s0, unsigned int ng)
{
    if (mdp == nullptr || mdp->goals != nullptr || mdp->S != nullptr ||
            mdp->T != nullptr || mdp->R != nullptr || n == 0 || ns == 0 || n < ns ||
            m == 0 || gamma < 0.0f || gamma > 1.0f || horizon == 0 || epsilon < 0.0f ||
            s0 >= n || ng >= n) {
        fprintf(stderr, "Error[mdp_initialize]: %s\n", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    mdp->n = n;
    mdp->ns = ns;
    mdp->m = m;
    mdp->gamma = gamma;
    mdp->horizon = horizon;
    mdp->epsilon = epsilon;
    mdp->s0 = s0;
    mdp->ng = ng;

    mdp->S = new int[mdp->n * mdp->m * mdp->ns];
    mdp->T = new float[mdp->n * mdp->m * mdp->ns];
    for (unsigned int i = 0; i < mdp->n * mdp->m * mdp->ns; i++) {
        mdp->S[i] = -1;
        mdp->T[i] = 0.0f;
    }

    mdp->R = new float[mdp->n * mdp->m];
    for (unsigned int i = 0; i < mdp->n * mdp->m; i++) {
        mdp->R[i] = 0.0f;
    }

    if (mdp->ng > 0) {
        mdp->goals = new unsigned int[ng];
        for (unsigned int i = 0; i < mdp->ng; i++) {
            mdp->goals[i] = 0;
        }
    } else {
        mdp->goals = nullptr;
    }

    mdp->d_goals = nullptr;
    mdp->d_S = nullptr;
    mdp->d_T = nullptr;
    mdp->d_R = nullptr;

    return NOVA_SUCCESS;
}


int mdp_uninitialize(MDP *mdp)
{
    if (mdp == nullptr) {
        fprintf(stderr, "Error[mdp_uninitialize]: %s\n", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    mdp->n = 0;
    mdp->ns = 0;
    mdp->m = 0;
    mdp->gamma = 0.0f;
    mdp->horizon = 0;
    mdp->epsilon = 0.0f;
    mdp->s0 = 0;
    mdp->ng = 0;

    if (mdp->goals != nullptr) {
        delete [] mdp->goals;
    }
    mdp->goals = nullptr;

    if (mdp->S != nullptr) {
        delete [] mdp->S;
    }
    mdp->S = nullptr;

    if (mdp->T != nullptr) {
        delete [] mdp->T;
    }
    mdp->T = nullptr;

    if (mdp->R != nullptr) {
        delete [] mdp->R;
    }
    mdp->R = nullptr;

    return NOVA_SUCCESS;
}

}; // namespace nova


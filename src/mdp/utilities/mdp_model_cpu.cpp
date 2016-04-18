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


#include "utilities/mdp_model_cpu.h"

#include <stdio.h>
#include <cstring>

#include "error_codes.h"
#include "constants.h"


namespace nova {

int mdp_uninitialize_cpu(MDP *mdp)
{
    if (mdp == nullptr) {
        fprintf(stderr, "Error[mdp_uninitialize_cpu]: %s\n", "Invalid input.");
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


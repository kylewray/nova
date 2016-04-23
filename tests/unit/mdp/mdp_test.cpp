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


#include "unit/mdp/mdp_test.h"

#include "mdp.h"
#include "error_codes.h"
#include "constants.h"


namespace nova {
namespace tests {

nova::MDP *create_simple_mdp()
{
    nova::MDP *mdp = new nova::MDP();

    mdp->n = 1;
    mdp->ns = 1;
    mdp->m = 1;

    mdp->S = new int[1];
    mdp->S[0] = 0;

    mdp->T = new float[1];
    mdp->T[0] = 1.0f;

    mdp->R = new float[1];
    mdp->R[0] = 1.0f;

    mdp->gamma = 0.9f;
    mdp->horizon = 3;
    mdp->epsilon = 0.0001f;

    mdp->ng = 1;
    mdp->goals = new unsigned int[1];
    mdp->goals[0] = 0;

    return mdp;
}

}; // namespace tests
}; // namespace nova


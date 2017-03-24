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


#include <nova/mdp/algorithms/ssp_lrtdp_cpu.h>

#include <gtest/gtest.h>

#include <nova/mdp/utilities/mdp_model_cpu.h>
#include <nova/error_codes.h>
#include <nova/constants.h>

#include <unit/mdp/mdp_tests.h>

namespace nova {
namespace tests {

TEST(SSPLRTDPCPU, initialization)
{
    srand(42); // Force random behavior to be deterministic.

    nova::MDP mdp;
    mdp.n = 2;
    mdp.m = 1337;

    nova::SSPLRTDPCPU lrtdp;
    lrtdp.VInitial = nullptr;
    lrtdp.trials = 100;

    int result = nova::ssp_lrtdp_initialize_cpu(&mdp, &lrtdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    EXPECT_EQ(lrtdp.VInitial, nullptr);
    EXPECT_EQ(lrtdp.currentHorizon, 0);

    EXPECT_NE(lrtdp.V, nullptr);
    if (lrtdp.V != nullptr) {
        EXPECT_NEAR(lrtdp.V[0], 0.0f, 1e-5f);
        EXPECT_NEAR(lrtdp.V[1], 0.0f, 1e-5f);
        delete [] lrtdp.V;
    }

    EXPECT_NE(lrtdp.pi, nullptr);
    if (lrtdp.pi != nullptr) {
        EXPECT_EQ(lrtdp.pi[0], 1337);
        EXPECT_EQ(lrtdp.pi[1], 1337);
        delete [] lrtdp.pi;
    }

    lrtdp.VInitial = new float[2];
    lrtdp.VInitial[0] = -1.0f;
    lrtdp.VInitial[1] = 1.0f;

    result = nova::ssp_lrtdp_initialize_cpu(&mdp, &lrtdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    EXPECT_EQ(lrtdp.currentHorizon, 0);

    EXPECT_NE(lrtdp.V, nullptr);
    if (lrtdp.V != nullptr) {
        EXPECT_NEAR(lrtdp.V[0], -1.0f, 1e-5f);
        EXPECT_NEAR(lrtdp.V[1], 1.0f, 1e-5f);
        delete [] lrtdp.V;
    }

    EXPECT_NE(lrtdp.pi, nullptr);
    if (lrtdp.pi != nullptr) {
        EXPECT_EQ(lrtdp.pi[0], 1337);
        EXPECT_EQ(lrtdp.pi[1], 1337);
        delete [] lrtdp.pi;
    }

    delete [] lrtdp.VInitial;
}


TEST(SSPLRTDPCPU, badInitializations)
{
    srand(42); // Force random behavior to be deterministic.

    nova::MDP mdp;
    mdp.n = 0;
    mdp.m = 0;

    nova::SSPLRTDPCPU lrtdp;
    lrtdp.VInitial = new float[2];
    lrtdp.VInitial[0] = -1.0f;
    lrtdp.VInitial[1] = 1.0f;
    lrtdp.trials = 100;

    int result = 0;

    bool requiresCleanup = false;

    result = nova::ssp_lrtdp_initialize_cpu(nullptr, &lrtdp);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    if (result != NOVA_ERROR_INVALID_DATA) {
        requiresCleanup = true;
    }

    result = nova::ssp_lrtdp_initialize_cpu(&mdp, nullptr);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    if (result != NOVA_ERROR_INVALID_DATA) {
        requiresCleanup = true;
    }

    result = nova::ssp_lrtdp_initialize_cpu(&mdp, &lrtdp);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    if (result != NOVA_ERROR_INVALID_DATA) {
        requiresCleanup = true;
    }

    if (requiresCleanup) {
        result = nova::ssp_lrtdp_uninitialize_cpu(&mdp, &lrtdp);
        EXPECT_EQ(result, NOVA_SUCCESS);
    }

    delete [] lrtdp.VInitial;
}


TEST(SSPLRTDPCPU, executionSimpleMDP)
{
    srand(42); // Force random behavior to be deterministic.

    nova::MDP *mdp = create_simple_mdp(true);

    nova::SSPLRTDPCPU lrtdp;
    lrtdp.VInitial = nullptr;
    lrtdp.trials = 100;

    nova::MDPValueFunction *policy = new nova::MDPValueFunction();

    int result = nova::ssp_lrtdp_execute_cpu(mdp, &lrtdp, policy);
    EXPECT_EQ(result, NOVA_SUCCESS);

    EXPECT_NE(policy, nullptr);

    if (policy != nullptr) {
        EXPECT_EQ(policy->n, 1);
        EXPECT_EQ(policy->m, 1);

        EXPECT_EQ(policy->r, 1);
        EXPECT_NE(policy->S, nullptr);
        if (policy->S != nullptr) {
            EXPECT_EQ(policy->S[0], 0);
        }

        // Note: The value (cost) is 3 because the horizon is 3; this is essentially 'max iterations'.
        // In the limit, this should be infinity.
        EXPECT_NE(policy->V, nullptr);
        if (policy->V != nullptr) {
            //EXPECT_EQ(policy->V[0], FLT_MAX);
            EXPECT_NEAR((double)policy->V[0], (double)FLT_MAX, (double)1.0);
        }

        EXPECT_NE(policy->pi, nullptr);
        if (policy->pi != nullptr) {
            EXPECT_EQ(policy->pi[0], 0);
        }
    }

    if (policy != nullptr) {
        result = nova::mdp_value_function_uninitialize(policy);
        EXPECT_EQ(result, NOVA_SUCCESS);
        delete policy;
        policy = nullptr;
    }

    result = nova::mdp_uninitialize_cpu(mdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    delete mdp;
}


TEST(SSPLRTDPCPU, executionThreeStateMDP)
{
    srand(42); // Force random behavior to be deterministic.

    nova::MDP *mdp = create_three_state_mdp(true);

    nova::SSPLRTDPCPU lrtdp;
    lrtdp.VInitial = nullptr;
    lrtdp.trials = 100;

    nova::MDPValueFunction *policy = new nova::MDPValueFunction();

    int result = nova::ssp_lrtdp_execute_cpu(mdp, &lrtdp, policy);
    EXPECT_EQ(result, NOVA_SUCCESS);

    EXPECT_NE(policy, nullptr);

    if (policy != nullptr) {
        EXPECT_EQ(policy->n, 3);
        EXPECT_EQ(policy->m, 2);

        EXPECT_EQ(policy->r, 3);
        EXPECT_NE(policy->S, nullptr);
        if (policy->S != nullptr) {
            EXPECT_EQ(policy->S[0], 0);
            EXPECT_EQ(policy->S[1], 1);
            EXPECT_EQ(policy->S[2], 2);
        }

        // Note: The value (cost) is 3.84375 because the horizon is 5; this is essentially 'max iterations'.
        // This value can be verified exactly:
        // 1+0.5*2+0.5*(1 + 0.5*2+0.5*(1 + 0.5*2+0.5*(1 + 0.5*2+0.5*(1 + 0.5*1+0.5*0)))) = 3.84375
        EXPECT_NE(policy->V, nullptr);
        if (policy->V != nullptr) {
            EXPECT_NEAR(policy->V[0], 4.0f, 1e-1f);
            EXPECT_NEAR(policy->V[1], 2.0f, 1e-1f);
            //EXPECT_NEAR(policy->V[2], 0.0f, 1e-1f);
        }

        EXPECT_NE(policy->pi, nullptr);
        if (policy->pi != nullptr) {
            EXPECT_EQ(policy->pi[0], 0);
            EXPECT_EQ(policy->pi[1], 1);
            //EXPECT_EQ(policy->pi[2], 0);
        }
    }

    if (policy != nullptr) {
        result = nova::mdp_value_function_uninitialize(policy);
        EXPECT_EQ(result, NOVA_SUCCESS);
        delete policy;
        policy = nullptr;
    }

    result = nova::mdp_uninitialize_cpu(mdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    delete mdp;
}


TEST(SSPLRTDPCPU, badExecution)
{
    srand(42); // Force random behavior to be deterministic.

    nova::MDP *mdp = create_simple_mdp(true);

    nova::SSPLRTDPCPU lrtdp;
    lrtdp.VInitial = nullptr;
    lrtdp.trials = 100;

    nova::MDPValueFunction *policy = nullptr;

    int result = nova::ssp_lrtdp_execute_cpu(nullptr, &lrtdp, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);

    result = nova::ssp_lrtdp_execute_cpu(mdp, nullptr, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);

    policy = new nova::MDPValueFunction();
    result = nova::ssp_lrtdp_execute_cpu(mdp, &lrtdp, nullptr);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);

    mdp->n = 0;
    result = nova::ssp_lrtdp_execute_cpu(mdp, &lrtdp, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->n = 1;

    mdp->ns = 0;
    result = nova::ssp_lrtdp_execute_cpu(mdp, &lrtdp, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->ns = 1;

    mdp->m = 0;
    result = nova::ssp_lrtdp_execute_cpu(mdp, &lrtdp, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->m = 1;

    void *tempAddress = nullptr;

    tempAddress = mdp->S;
    mdp->S = nullptr;
    result = nova::ssp_lrtdp_execute_cpu(mdp, &lrtdp, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->S = (int *)tempAddress;

    tempAddress = mdp->T;
    mdp->T = nullptr;
    result = nova::ssp_lrtdp_execute_cpu(mdp, &lrtdp, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->T = (float *)tempAddress;

    tempAddress = mdp->R;
    mdp->R = nullptr;
    result = nova::ssp_lrtdp_execute_cpu(mdp, &lrtdp, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->R = (float *)tempAddress;

    mdp->horizon = 0;
    result = nova::ssp_lrtdp_execute_cpu(mdp, &lrtdp, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->horizon = 3;

    mdp->ng = 0;
    result = nova::ssp_lrtdp_execute_cpu(mdp, &lrtdp, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->ng = 1;

    tempAddress = mdp->goals;
    mdp->goals = nullptr;
    result = nova::ssp_lrtdp_execute_cpu(mdp, &lrtdp, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->goals = (unsigned int *)tempAddress;

    if (policy != nullptr) {
        result = nova::mdp_value_function_uninitialize(policy);
        EXPECT_EQ(result, NOVA_SUCCESS);
        delete policy;
        policy = nullptr;
    }

    result = nova::mdp_uninitialize_cpu(mdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    delete mdp;
}


TEST(SSPLRTDPCPU, uninitialization)
{
    srand(42); // Force random behavior to be deterministic.

    nova::MDP mdp;

    nova::SSPLRTDPCPU lrtdp;
    lrtdp.VInitial = nullptr;
    lrtdp.trials = 100;
    lrtdp.V = new float[1];
    lrtdp.pi = new unsigned int[1];

    int result = nova::ssp_lrtdp_uninitialize_cpu(&mdp, &lrtdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    EXPECT_EQ(lrtdp.currentHorizon, 0);
    EXPECT_EQ(lrtdp.V, nullptr);
    EXPECT_EQ(lrtdp.pi, nullptr);

    if (lrtdp.V != nullptr) {
        delete [] lrtdp.V;
    }
    if (lrtdp.pi != nullptr) {
        delete [] lrtdp.pi;
    }
}


TEST(SSPLRTDPCPU, badUninitialization)
{
    srand(42); // Force random behavior to be deterministic.

    nova::MDP mdp;

    int result = nova::ssp_lrtdp_uninitialize_cpu(nullptr, nullptr);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);

    result = nova::ssp_lrtdp_uninitialize_cpu(&mdp, nullptr);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
}


// For performance reasons, there is no checking here, so we simply
// are checking valid mathematics.
TEST(SSPLRTDPCPU, updateSimpleMDP)
{
    srand(42); // Force random behavior to be deterministic.

    nova::MDP *mdp = create_simple_mdp(true);

    nova::SSPLRTDPCPU lrtdp;
    lrtdp.VInitial = nullptr;
    lrtdp.trials = 100;

    int result = nova::ssp_lrtdp_initialize_cpu(mdp, &lrtdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    mdp->horizon = 1;
    lrtdp.currentHorizon = 0;
    lrtdp.pi[0] = 1337;

    result = nova::ssp_lrtdp_update_cpu(mdp, &lrtdp);
    EXPECT_EQ(result == NOVA_SUCCESS || result == NOVA_CONVERGED, true);
    EXPECT_EQ(lrtdp.currentHorizon, 0);
    EXPECT_NEAR((double)lrtdp.V[0], (double)FLT_MAX, (double)1.0);
    //EXPECT_EQ(lrtdp.V[0], FLT_MAX);
    EXPECT_EQ(lrtdp.pi[0], 0);

    mdp->horizon = 2;
    lrtdp.pi[0] = 1337;

    result = nova::ssp_lrtdp_update_cpu(mdp, &lrtdp);
    EXPECT_EQ(result == NOVA_SUCCESS || result == NOVA_CONVERGED, true);
    EXPECT_EQ(lrtdp.currentHorizon, 0);
    EXPECT_NEAR((double)lrtdp.V[0], (double)FLT_MAX, (double)1.0);
    //EXPECT_EQ(lrtdp.V[0], FLT_MAX);
    EXPECT_EQ(lrtdp.pi[0], 0);

    result = nova::ssp_lrtdp_uninitialize_cpu(mdp, &lrtdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    result = nova::mdp_uninitialize_cpu(mdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    delete mdp;
}


TEST(SSPLRTDPCPU, updateThreeStateMDP)
{
    srand(42); // Force random behavior to be deterministic.

    nova::MDP *mdp = create_three_state_mdp(true);

    nova::SSPLRTDPCPU lrtdp;
    lrtdp.VInitial = nullptr;
    lrtdp.trials = 100;

    int result = nova::ssp_lrtdp_initialize_cpu(mdp, &lrtdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    mdp->horizon = 1;
    lrtdp.currentHorizon = 0;
    lrtdp.pi[0] = 1337;
    lrtdp.pi[1] = 1337;
    lrtdp.pi[2] = 1337;

    result = nova::ssp_lrtdp_update_cpu(mdp, &lrtdp);
    EXPECT_EQ(result == NOVA_SUCCESS || result == NOVA_CONVERGED, true);
    EXPECT_LE(lrtdp.currentHorizon, 1);
    EXPECT_NEAR(lrtdp.V[0], 1.0f, 1e-1f);
    EXPECT_NEAR(lrtdp.V[1], 0.0f, 1e-5f);
    EXPECT_NEAR(lrtdp.V[2], 0.0f, 1e-5f);
    EXPECT_EQ(lrtdp.pi[0], 0);
    EXPECT_EQ(lrtdp.pi[1], 1337);
    EXPECT_EQ(lrtdp.pi[2], 1337);

    mdp->horizon = 2;
    lrtdp.pi[0] = 1337;
    lrtdp.pi[1] = 1337;
    lrtdp.pi[2] = 1337;

    result = nova::ssp_lrtdp_update_cpu(mdp, &lrtdp);
    EXPECT_EQ(result == NOVA_SUCCESS || result == NOVA_CONVERGED, true);
    EXPECT_LE(lrtdp.currentHorizon, 2);
    EXPECT_NEAR(lrtdp.V[0], 1.75f, 1e-1f);
    EXPECT_NEAR(lrtdp.V[1], 0.0f, 1e-5f);
    EXPECT_NEAR(lrtdp.V[2], 0.0f, 1e-5f);
    EXPECT_EQ(lrtdp.pi[0], 0);
    EXPECT_EQ(lrtdp.pi[1], 0);
    EXPECT_EQ(lrtdp.pi[2], 1337);

    mdp->horizon = 5;
    lrtdp.pi[0] = 1337;
    lrtdp.pi[1] = 1337;
    lrtdp.pi[2] = 1337;

    result = nova::ssp_lrtdp_update_cpu(mdp, &lrtdp);
    EXPECT_EQ(result == NOVA_SUCCESS || result == NOVA_CONVERGED, true);
    EXPECT_LE(lrtdp.currentHorizon, 5);
    EXPECT_NEAR(lrtdp.V[0], 1.984375f, 1e-1f);
    EXPECT_NEAR(lrtdp.V[1], 1.0f, 1e-1f);
    EXPECT_NEAR(lrtdp.V[2], 0.0f, 1e-5f);
    EXPECT_EQ(lrtdp.pi[0], 0);
    EXPECT_EQ(lrtdp.pi[1], 0);
    //EXPECT_EQ(lrtdp.pi[2], 0);

    mdp->horizon = 10;
    lrtdp.pi[0] = 1337;
    lrtdp.pi[1] = 1337;
    lrtdp.pi[2] = 1337;

    result = nova::ssp_lrtdp_update_cpu(mdp, &lrtdp);
    EXPECT_EQ(result == NOVA_SUCCESS || result == NOVA_CONVERGED, true);
    EXPECT_LE(lrtdp.currentHorizon, 10);
    EXPECT_NEAR(lrtdp.V[0], 2.93652f, 1e-1f);
    EXPECT_NEAR(lrtdp.V[1], 2.0f, 1e-1f);
    EXPECT_NEAR(lrtdp.V[2], 0.0f, 1e-5f);
    EXPECT_EQ(lrtdp.pi[0], 0);
    EXPECT_EQ(lrtdp.pi[1], 1);
    //EXPECT_EQ(lrtdp.pi[2], 0);

    result = nova::ssp_lrtdp_uninitialize_cpu(mdp, &lrtdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    result = nova::mdp_uninitialize_cpu(mdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    delete mdp;
}


// There is no bad update because for performance reasons, we do not
// check or sanitize input to this function.
//TEST(SSPLRTDPCPU, badUpdate) { }


TEST(SSPLRTDPCPU, getPolicy)
{
    srand(42); // Force random behavior to be deterministic.

    nova::MDP mdp;
    mdp.n = 2;
    mdp.m = 100;

    nova::SSPLRTDPCPU lrtdp;
    lrtdp.VInitial = nullptr;
    lrtdp.trials = 100;

    lrtdp.V = new float[2];
    lrtdp.V[0] = 10.0f;
    lrtdp.V[1] = 20.0f;

    lrtdp.pi = new unsigned int[2];
    lrtdp.pi[0] = 50;
    lrtdp.pi[1] = 60;

    nova::MDPValueFunction *policy = new nova::MDPValueFunction();

    int result = nova::ssp_lrtdp_get_policy_cpu(&mdp, &lrtdp, policy);
    EXPECT_EQ(result, NOVA_SUCCESS);

    EXPECT_NE(policy, nullptr);

    if (policy != nullptr) {
        EXPECT_EQ(policy->n, mdp.n);
        EXPECT_EQ(policy->m, mdp.m);

        EXPECT_EQ(policy->r, 2);
        EXPECT_NE(policy->S, nullptr);
        if (policy->S != nullptr) {
            EXPECT_EQ(policy->S[0], 0);
            EXPECT_EQ(policy->S[1], 1);
        }

        EXPECT_NE(policy->V, nullptr);
        if (policy->V != nullptr) {
            EXPECT_NEAR(policy->V[0], 10.0f, 1e-5f);
            EXPECT_NEAR(policy->V[1], 20.0f, 1e-5f);
        }

        EXPECT_NE(policy->pi, nullptr);
        if (policy->pi != nullptr) {
            EXPECT_EQ(policy->pi[0], 50);
            EXPECT_EQ(policy->pi[1], 60);
        }

        result = nova::mdp_value_function_uninitialize(policy);
        EXPECT_EQ(result, NOVA_SUCCESS);
        delete policy;
        policy = nullptr;
    }

    delete [] lrtdp.V;
    delete [] lrtdp.pi;
}


TEST(SSPLRTDPCPU, badGetPolicy)
{
    srand(42); // Force random behavior to be deterministic.

    nova::MDP mdp;

    nova::SSPLRTDPCPU lrtdp;
    lrtdp.VInitial = nullptr;
    lrtdp.trials = 100;

    int result = nova::ssp_lrtdp_get_policy_cpu(nullptr, nullptr, nullptr);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);

    result = nova::ssp_lrtdp_get_policy_cpu(&mdp, nullptr, nullptr);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);

    result = nova::ssp_lrtdp_get_policy_cpu(&mdp, &lrtdp, nullptr);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
}

}; // namespace tests
}; // namespace nova




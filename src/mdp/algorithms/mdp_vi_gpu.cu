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


#include "algorithms/mdp_vi_gpu.h"

#include <cmath>
#include <stdio.h>

#include "utilities/mdp_model_gpu.h"
#include "error_codes.h"
#include "constants.h"

namespace nova {

__global__ void mdp_vi_bellman_update_gpu(unsigned int n, unsigned int ns, unsigned int m, float gamma,
        const int *S, const float *T, const float *R, const float *V,
        float *VPrime, unsigned int *pi)
{
    // The current state as a function of the blocks and threads.
    int s;

    // The intermediate Q(s, a) value.
    float Qsa;

    // The index within S and T (i.e., in n*s*ns).
    int index;

    // The true successor state index (in 0 to n-1), resolved using S.
    int spindex;

    // Compute the index of the state. Return if it is beyond the state.
    s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) {
        return;
    }

    // Nvidia GPUs follow IEEE floating point standards, so this should be safe.
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

        __syncthreads();

        if (a == 0 || Qsa > VPrime[s]) {
            VPrime[s] = Qsa;
            pi[s] = a;
        }

        __syncthreads();
    }
}


int mdp_vi_initialize_gpu(const MDP *mdp, MDPVIGPU *vi)
{
    if (mdp == nullptr || mdp->n == 0 || vi == nullptr) {
        fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Reset the current horizon.
    vi->currentHorizon = 0;

    // Create VInitial if undefined.
    bool createdVInitial = false;
    if (vi->VInitial == nullptr) {
        vi->VInitial = new float[mdp->n];
        for (unsigned int i = 0; i < mdp->n; i++) {
            vi->VInitial[i] = 0.0f;
        }
        createdVInitial = true;
    }

    // Create the device-side V.
    if (cudaMalloc(&vi->d_V, mdp->n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n",
                "Failed to allocate device-side memory for the value function.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }
    if (cudaMemcpy(vi->d_V, vi->VInitial, mdp->n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n",
                "Failed to copy memory from host to device for the value function.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    if (cudaMalloc(&vi->d_VPrime, mdp->n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n",
                "Failed to allocate device-side memory for the value function (prime).");
        return NOVA_ERROR_DEVICE_MALLOC;
    }
    if (cudaMemcpy(vi->d_VPrime, vi->VInitial, mdp->n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n",
                "Failed to copy memory from host to device for the value function (prime).");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    // Create the device-side pi.
    if (cudaMalloc(&vi->d_pi, mdp->n * sizeof(unsigned int)) != cudaSuccess) {
        fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n",
                "Failed to allocate device-side memory for the policy (pi).");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // If we created VInitial, then free it.
    if (createdVInitial) {
        delete [] vi->VInitial;
        vi->VInitial = nullptr;
    }

    return NOVA_SUCCESS;
}


int mdp_vi_execute_gpu(const MDP *mdp, MDPVIGPU *vi, MDPValueFunction *&policy)
{
    // The result from calling other functions.
    int result;

    // First, ensure data is valid.
    if (mdp == nullptr || mdp->n == 0 || mdp->ns == 0 || mdp->m == 0 ||
            mdp->S == nullptr || mdp->T == nullptr || mdp->R == nullptr ||
            mdp->gamma < 0.0f || mdp->gamma > 1.0f || mdp->horizon < 1 ||
            vi == nullptr || policy != nullptr) {
        fprintf(stderr, "Error[mdp_vi_execute_gpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Ensure threads are correct.
    if (vi->numThreads % 32 != 0) {
        fprintf(stderr, "Error[mdp_vi_execute_gpu]: %s\n", "Invalid number of threads.");
        return NOVA_ERROR_INVALID_CUDA_PARAM;
    }

    result = mdp_vi_initialize_gpu(mdp, vi);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[mdp_vi_execute_gpu]: %s\n", "Failed to initialize GPU variables.");
        return result;
    }

    // We iterate over all time steps up to the horizon. Initialize set the currentHorizon to 0,
    // and the update increments it.
    while (vi->currentHorizon < mdp->horizon) {
        result = mdp_vi_update_gpu(mdp, vi);
        if (result != NOVA_SUCCESS) {
            fprintf(stderr, "Error[mdp_vi_execute_gpu]: %s\n", "Failed to perform Bellman update on the GPU.");
            return result;
        }
    }

    result = mdp_vi_get_policy_gpu(mdp, vi, policy);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[mdp_vi_execute_gpu]: %s\n", "Failed to get the policy.");
        return result;
    }

    result = mdp_vi_uninitialize_gpu(mdp, vi);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[mdp_vi_execute_gpu]: %s\n", "Failed to uninitialize the GPU variables.");
        return result;
    }

    return NOVA_SUCCESS;
}


int mdp_vi_uninitialize_gpu(const MDP *mdp, MDPVIGPU *vi)
{
    if (mdp == nullptr || vi == nullptr) {
        fprintf(stderr, "Error[mdp_vi_uninitialize_gpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    int result;

    result = NOVA_SUCCESS;

    // Reset the current horizon.
    vi->currentHorizon = 0;

    if (vi->d_V != nullptr) {
        if (cudaFree(vi->d_V) != cudaSuccess) {
            fprintf(stderr, "Error[mdp_vi_uninitialize_gpu]: %s\n",
                    "Failed to free memory from device for the value function.");
            result = NOVA_ERROR_DEVICE_FREE;
        }
    }
    vi->d_V = nullptr;

    if (vi->d_VPrime != nullptr) {
        if (cudaFree(vi->d_VPrime) != cudaSuccess) {
            fprintf(stderr, "Error[mdp_vi_uninitialize_gpu]: %s\n",
                    "Failed to free memory from device for the value function (prime).");
            result = NOVA_ERROR_DEVICE_FREE;
        }
    }
    vi->d_VPrime = nullptr;

    if (vi->d_pi != nullptr) {
        if (cudaFree(vi->d_pi) != cudaSuccess) {
            fprintf(stderr, "Error[mdp_vi_uninitialize_gpu]: %s\n",
                    "Failed to free memory from device for the policy (pi).");
            result = NOVA_ERROR_DEVICE_FREE;
        }
    }
    vi->d_pi = nullptr;

    return result;
}


int mdp_vi_update_gpu(const MDP *mdp, MDPVIGPU *vi)
{
    unsigned int numBlocks;

    // Compute the number of blocks.
    numBlocks = (unsigned int)((float)mdp->n / (float)vi->numThreads) + 1;

    // Execute value iteration for these number of iterations. For each iteration, however,
    // we will run the state updates in parallel.
    if (vi->currentHorizon % 2 == 0) {
        mdp_vi_bellman_update_gpu<<< numBlocks, vi->numThreads >>>(
                    mdp->n, mdp->ns, mdp->m, mdp->gamma,
                    mdp->d_S, mdp->d_T, mdp->d_R,
                    vi->d_V, vi->d_VPrime, vi->d_pi);
    } else {
        mdp_vi_bellman_update_gpu<<< numBlocks, vi->numThreads >>>(
                    mdp->n, mdp->ns, mdp->m, mdp->gamma,
                    mdp->d_S, mdp->d_T, mdp->d_R,
                    vi->d_VPrime, vi->d_V, vi->d_pi);
    }

    // Check if there was an error executing the kernel.
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Error[mdp_vi_update_gpu]: %s\n",
                        "Failed to execute the 'Bellman update' kernel.");
        return NOVA_ERROR_KERNEL_EXECUTION;
    }

    // Wait for the kernel to finish before looping more.
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "Error[mdp_vi_update_gpu]: %s\n",
                    "Failed to synchronize the device after 'Bellman update' kernel.");
        return NOVA_ERROR_DEVICE_SYNCHRONIZE;
    }

    vi->currentHorizon++;

    return NOVA_SUCCESS;
}


int mdp_vi_get_policy_gpu(const MDP *mdp, MDPVIGPU *vi, MDPValueFunction *&policy)
{
    if (mdp == nullptr || vi == nullptr || policy != nullptr) {
        fprintf(stderr, "Error[mdp_vi_get_policy_gpu]: %s\n",
                        "Invalid arguments. The policy must be undefined.");
        return NOVA_ERROR_INVALID_DATA;
    }

    policy = new MDPValueFunction();

    policy->n = mdp->n;
    policy->m = mdp->m;
    policy->r = 0;

    policy->S = nullptr;
    policy->V = new float[mdp->n];
    policy->pi = new unsigned int[mdp->n];

    // Copy the final (or intermediate) result, both V and pi, from device to host. This assumes
    // that the memory has been allocated for the variables provided.
    if (vi->currentHorizon % 2 == 0) {
        if (cudaMemcpy(policy->V, vi->d_V, mdp->n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "Error[mdp_vi_get_policy_gpu]: %s\n",
                    "Failed to copy memory from device to host for the value function.");
            return NOVA_ERROR_MEMCPY_TO_HOST;
        }
    } else {
        if (cudaMemcpy(policy->V, vi->d_VPrime, mdp->n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "Error[mdp_vi_get_policy_gpu]: %s\n",
                    "Failed to copy memory from device to host for the value function (prime).");
            return NOVA_ERROR_MEMCPY_TO_HOST;
        }
    }
    if (cudaMemcpy(policy->pi, vi->d_pi, mdp->n * sizeof(unsigned int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Error[mdp_vi_get_policy_gpu]: %s\n",
                "Failed to copy memory from device to host for the policy (pi).");
        return NOVA_ERROR_MEMCPY_TO_HOST;
    }

    return NOVA_SUCCESS;
}

}; // namespace nova


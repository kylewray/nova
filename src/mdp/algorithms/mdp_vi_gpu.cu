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


#include "algorithms/mdp_vi_gpu.h"
#include "utilities/mdp_model_gpu.h"
#include "error_codes.h"
#include "constants.h"

#include <cmath>
#include <stdio.h>


__global__ void mdp_bellman_update_gpu(unsigned int n, unsigned int ns, unsigned int m, float gamma,
        const int *S, const float *T, const float *R, const float *V, float *VPrime, unsigned int *pi)
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


int mdp_vi_complete_gpu(MDP *mdp, unsigned int numThreads, const float *Vinitial, float *&V, unsigned int *&pi)
{
    int result;

    result = mdp_initialize_successors_gpu(mdp);
    if (result != NOVA_SUCCESS) {
        return result;
    }
    result = mdp_initialize_state_transitions_gpu(mdp);
    if (result != NOVA_SUCCESS) {
        return result;
    }
    result = mdp_initialize_rewards_gpu(mdp);
    if (result != NOVA_SUCCESS) {
        return result;
    }

    result = mdp_vi_execute_gpu(mdp, numThreads, Vinitial, V, pi);
    if (result != NOVA_SUCCESS) {
        return result;
    }

    result = NOVA_SUCCESS;
    if (mdp_uninitialize_successors_gpu(mdp) != NOVA_SUCCESS) {
        result = NOVA_ERROR_DEVICE_FREE;
    }
    if (mdp_uninitialize_state_transitions_gpu(mdp) != NOVA_SUCCESS) {
        result = NOVA_ERROR_DEVICE_FREE;
    }
    if (mdp_uninitialize_rewards_gpu(mdp) != NOVA_SUCCESS) {
        result = NOVA_ERROR_DEVICE_FREE;
    }

    return result;
}


int mdp_vi_initialize_gpu(MDP *mdp, const float *Vinitial)
{
    // Reset the current horizon.
    mdp->currentHorizon = 0;

    // Create the device-side V.
    if (cudaMalloc(&mdp->d_V, mdp->n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n",
                "Failed to allocate device-side memory for the value function.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }
    if (cudaMemcpy(mdp->d_V, Vinitial, mdp->n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n",
                "Failed to copy memory from host to device for the value function.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    if (cudaMalloc(&mdp->d_VPrime, mdp->n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n",
                "Failed to allocate device-side memory for the value function (prime).");
        return NOVA_ERROR_DEVICE_MALLOC;
    }
    if (cudaMemcpy(mdp->d_VPrime, Vinitial, mdp->n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n",
                "Failed to copy memory from host to device for the value function (prime).");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    // Create the device-side pi.
    if (cudaMalloc(&mdp->d_pi, mdp->n * sizeof(unsigned int)) != cudaSuccess) {
        fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n",
                "Failed to allocate device-side memory for the policy (pi).");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    return NOVA_SUCCESS;
}


int mdp_vi_execute_gpu(MDP *mdp, unsigned int numThreads, const float *Vinitial, float *&V, unsigned int *&pi)
{
    // The result from calling other functions.
    int result;

    // First, ensure data is valid.
    if (mdp->n == 0 || mdp->ns == 0 || mdp->m == 0 ||
            mdp->S == nullptr || mdp->T == nullptr || mdp->R == nullptr ||
            mdp->gamma < 0.0f || mdp->gamma > 1.0f || mdp->horizon < 1 ||
            Vinitial == nullptr || V != nullptr || pi != nullptr) {
        fprintf(stderr, "Error[mdp_vi_execute_gpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Ensure threads are correct.
    if (numThreads % 32 != 0) {
        fprintf(stderr, "Error[mdp_vi_execute_gpu]: %s\n", "Invalid number of threads.");
        return NOVA_ERROR_INVALID_CUDA_PARAM;
    }

    result = mdp_vi_initialize_gpu(mdp, Vinitial);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[mdp_vi_execute_gpu]: %s\n", "Failed to initialize GPU variables.");
        return result;
    }

    // We iterate over all time steps up to the horizon. Initialize set the currentHorizon to 0,
    // and the update increments it.
    while (mdp->currentHorizon < mdp->horizon) {
        result = mdp_vi_update_gpu(mdp, numThreads);
        if (result != NOVA_SUCCESS) {
            fprintf(stderr, "Error[mdp_vi_execute_gpu]: %s\n", "Failed to perform Bellman update on the GPU.");
            return result;
        }
    }

    result = mdp_vi_get_policy_gpu(mdp, V, pi);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[mdp_vi_execute_gpu]: %s\n", "Failed to get the policy.");
        return result;
    }

    result = mdp_vi_uninitialize_gpu(mdp);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[mdp_vi_execute_gpu]: %s\n", "Failed to uninitialize the GPU variables.");
        return result;
    }

    return NOVA_SUCCESS;
}


int mdp_vi_uninitialize_gpu(MDP *mdp)
{
    int result;

    result = NOVA_SUCCESS;

    // Reset the current horizon.
    mdp->currentHorizon = 0;

    if (mdp->d_V != nullptr) {
        if (cudaFree(mdp->d_V) != cudaSuccess) {
            fprintf(stderr, "Error[mdp_vi_uninitialize_gpu]: %s\n",
                    "Failed to free memory from device for the value function.");
            result = NOVA_ERROR_DEVICE_FREE;
        }
    }
    mdp->d_V = nullptr;

    if (mdp->d_VPrime != nullptr) {
        if (cudaFree(mdp->d_VPrime) != cudaSuccess) {
            fprintf(stderr, "Error[mdp_vi_uninitialize_gpu]: %s\n",
                    "Failed to free memory from device for the value function (prime).");
            result = NOVA_ERROR_DEVICE_FREE;
        }
    }
    mdp->d_VPrime = nullptr;

    if (mdp->d_pi != nullptr) {
        if (cudaFree(mdp->d_pi) != cudaSuccess) {
            fprintf(stderr, "Error[mdp_vi_uninitialize_gpu]: %s\n",
                    "Failed to free memory from device for the policy (pi).");
            result = NOVA_ERROR_DEVICE_FREE;
        }
    }
    mdp->d_pi = nullptr;

    return result;
}


int mdp_vi_update_gpu(MDP *mdp, unsigned int numThreads)
{
    unsigned int numBlocks;

    // Compute the number of blocks.
    numBlocks = (unsigned int)((float)mdp->n / (float)numThreads) + 1;

    // Execute value iteration for these number of iterations. For each iteration, however,
    // we will run the state updates in parallel.
    if (mdp->currentHorizon % 2 == 0) {
        mdp_bellman_update_gpu<<< numBlocks, numThreads >>>(
                    mdp->n, mdp->ns, mdp->m, mdp->gamma,
                    mdp->d_S, mdp->d_T, mdp->d_R,
                    mdp->d_V, mdp->d_VPrime, mdp->d_pi);
    } else {                                                                                     
        mdp_bellman_update_gpu<<< numBlocks, numThreads >>>(
                    mdp->n, mdp->ns, mdp->m, mdp->gamma,
                    mdp->d_S, mdp->d_T, mdp->d_R,
                    mdp->d_VPrime, mdp->d_V, mdp->d_pi);
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

    mdp->currentHorizon++;

    return NOVA_SUCCESS;
}


int mdp_vi_get_policy_gpu(MDP *mdp, float *&V, unsigned int *&pi)
{
    if (V != nullptr || pi != nullptr) {
        fprintf(stderr, "Error[mdp_vi_get_policy_gpu]: %s\n", "Invalid arguments. V and pi must be undefined.");
        return NOVA_ERROR_INVALID_DATA;
    }

    V = new float[mdp->n];
    pi = new unsigned int[mdp->n];

    // Copy the final (or intermediate) result, both V and pi, from device to host. This assumes
    // that the memory has been allocated for the variables provided.
    if (mdp->currentHorizon % 2 == 0) {
        if (cudaMemcpy(V, mdp->d_V, mdp->n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "Error[mdp_vi_get_policy_gpu]: %s\n",
                    "Failed to copy memory from device to host for the value function.");
            return NOVA_ERROR_MEMCPY_TO_HOST;
        }
    } else {
        if (cudaMemcpy(V, mdp->d_VPrime, mdp->n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "Error[mdp_vi_get_policy_gpu]: %s\n",
                    "Failed to copy memory from device to host for the value function (prime).");
            return NOVA_ERROR_MEMCPY_TO_HOST;
        }
    }
    if (cudaMemcpy(pi, mdp->d_pi, mdp->n * sizeof(unsigned int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Error[mdp_vi_get_policy_gpu]: %s\n",
                "Failed to copy memory from device to host for the policy (pi).");
        return NOVA_ERROR_MEMCPY_TO_HOST;
    }

    return NOVA_SUCCESS;
}


extern "C" int mdp_vi_free_policy_gpu(MDP *mdp, float *&V, unsigned int *&pi)
{
    if (V != nullptr) {
        delete [] V;
    }
    V = nullptr;

    if (pi != nullptr) {
        delete [] pi;
    }
    pi = nullptr;

    return NOVA_SUCCESS;
}


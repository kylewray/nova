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


#include "pomdp_pbvi_gpu.h"
#include "pomdp_model_gpu.h"
#include "error_codes.h"

#include <stdio.h>


// This is determined by hardware, so what is below is a 'safe' guess. If this is off, the
// program might return 'nan' or 'inf'. These come from IEEE floating-point standards.
#define FLT_MAX 1e+35
#define FLT_MIN -1e+35
#define FLT_ERR_TOL 1e-9


__global__ void pomdp_pbvi_initialize_alphaBA_gpu(unsigned int n, unsigned int m,
    unsigned int r, const float *R, float *alphaBA)
{
    unsigned int beliefIndex = blockIdx.x;
    unsigned int action = blockIdx.y;

    if (beliefIndex >= r || action >= m) {
        return;
    }

    // Compute Gamma_{a,*} and set it to the first value of alphaBA. Stride here.
    for (unsigned int s = threadIdx.x; s < n; s += blockDim.x) {
        alphaBA[beliefIndex * m * n + action * n + s] = R[s * m + action];
    }
}


__global__ void pomdp_pbvi_compute_alphaBA_gpu(unsigned int n, unsigned int m,
        unsigned int z, unsigned int r, unsigned int maxNonZeroBeliefs, unsigned int maxSuccessors,
        const float *B, const float *T, const float *O, const float *R,
        const bool *available, const int *nonZeroBeliefs, const int *successors,
        float gamma, const float *Gamma, const unsigned int *pi, float *alphaBA)
{
    // Since float and unsigned int are 4 bytes each, and we need each array to be the size of
    // the number of threads, we will need to call this with:
    // sizeof(float) * numThreads + sizeof(unsigned int * numThreads.
    // Note: blockDim.x == numThreads
    extern __shared__ float sdata[];
    float *maxAlphaDotBeta = (float *)sdata;
    unsigned int *maxAlphaIndex = (unsigned int *)&maxAlphaDotBeta[blockDim.x];

    maxAlphaDotBeta[threadIdx.x] = FLT_MIN;
    maxAlphaIndex[threadIdx.x] = 0;

    __syncthreads();

    unsigned int beliefIndex = blockIdx.x;
    unsigned int action = blockIdx.y;
    unsigned int observation = blockIdx.z;

    if (beliefIndex >= r || action >= m || observation >= z) {
        return;
    }

    // Compute the max alpha vector from Gamma, given the fixed action and observation.
    // Note: this is the max w.r.t. just the strided elements. The reduction will
    // be computed afterwards for the max over all alpha-vectors.

    for (unsigned int alphaIndex = threadIdx.x; alphaIndex < r; alphaIndex += blockDim.x) {
        float alphaDotBeta = 0.0f;

        for (unsigned int i = 0; i < maxNonZeroBeliefs; i++) {
            int s = nonZeroBeliefs[beliefIndex * maxNonZeroBeliefs + i];
            if (s < 0) {
                break;
            }

            // We compute the value of this state in the alpha-vector, then multiply it by the
            // belief, and add it to the current dot product value for this alpha-vector.
            float value = 0.0f;
            for (unsigned int j = 0; j < maxSuccessors; j++) {
                int sp = successors[s * m * maxSuccessors + action * maxSuccessors + j];
                if (sp < 0) {
                    break;
                }
                value += T[s * m * n + action * n + sp] *
                            O[action * n * z + sp * z + observation] *
                            Gamma[alphaIndex * n + sp];
            }
            value *= gamma;

            __syncthreads();

            alphaDotBeta += value * B[beliefIndex * n + s];
        }

        __syncthreads();

        // Store the maximal value and index.
        if (alphaIndex == threadIdx.x || alphaDotBeta > maxAlphaDotBeta[threadIdx.x]) {
            maxAlphaDotBeta[threadIdx.x] = alphaDotBeta;
            maxAlphaIndex[threadIdx.x] = alphaIndex;
        }
    }

    // Note: The above code essentially does the first add during load. It takes care of *all*
    // the other elements *outside* the number of threads we have. In other words, starting here,
    // we already have computed part of the maxAlphaDotBeta and maxAlphaIndex; we just need to
    // finish the rest quickly, using a reduction.
    __syncthreads();

    // Use reduction to compute the max overall alpha-vector.
    for (unsigned int alphaIndex = blockDim.x / 2; alphaIndex > 0; alphaIndex >>= 1) {
        if (threadIdx.x < alphaIndex && threadIdx.x < r && threadIdx.x + alphaIndex < r) {
            if (maxAlphaDotBeta[threadIdx.x] < maxAlphaDotBeta[threadIdx.x + alphaIndex]) {
                maxAlphaDotBeta[threadIdx.x] = maxAlphaDotBeta[threadIdx.x + alphaIndex];
                maxAlphaIndex[threadIdx.x] = maxAlphaIndex[threadIdx.x + alphaIndex];
            }
        }

        __syncthreads();
    }

    // Now we can compute the alpha-vector component for this observation, since we have the max.
    // We will need to compute the dot product anyway, so let's just distribute the belief over the
    // sum over observations, and add it all up here.
    // Note: This re-uses the thread to stride over states now.
    for (unsigned int s = threadIdx.x; s < n; s += blockDim.x) {
        // We compute the value of this state in the alpha-vector, then multiply it by the belief,
        // and add it to the current dot product value for this alpha-vector.
        float value = 0.0f;
        for (unsigned int i = 0; i < maxSuccessors; i++) {
            int sp = successors[s * m * maxSuccessors + action * maxSuccessors + i];
            if (sp < 0) {
                break;
            }
            // Note: maxAlphaIndex[0] holds the maximal index value computed from the reduction.
            value += T[s * m * n + action * n + sp] *
                        O[action * n * z + sp * z + observation] *
                        Gamma[maxAlphaIndex[0] * n + sp];
        }

        __syncthreads();

        alphaBA[beliefIndex * m * n + action * n + s] += gamma * value;
    }
}


__global__ void pomdp_pbvi_update_step_gpu(unsigned int n, unsigned int m, unsigned int z,
        unsigned int r, unsigned int maxNonZeroBeliefs, unsigned int maxSuccessors,
        const float *B, const float *T, const float *O, const float *R,
        const bool *available, const int *nonZeroBeliefs, const int *successors,
        float gamma, const float *Gamma, const unsigned int *pi,
        float *alphaBA, float *GammaPrime, unsigned int *piPrime)
{
    // Each block will run a different belief. Our overall goal: Compute the value
    // of GammaPrime[beliefIndex * n + ???] and piPrime[beliefIndex].
    unsigned int beliefIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (beliefIndex >= r) {
        return;
    }

    // We want to find the action that maximizes the value, store it in piPrime, as well as
    // its alpha-vector GammaPrime.
    float maxActionValue = FLT_MIN;

    for (unsigned int action = 0; action < m; action++) {
        // Only execute if the action is available.
        if (available[beliefIndex * m + action]) {
            // The potential alpha-vector has been computed, so compute the value with respect
            // to the belief state.
            float actionValue = 0.0f;
            for (unsigned int i = 0; i < maxNonZeroBeliefs; i++) {
                int s = nonZeroBeliefs[beliefIndex * maxNonZeroBeliefs + i];
                if (s < 0) {
                    break;
                }
                actionValue += alphaBA[beliefIndex * m * n + action * n + s] *
                                    B[beliefIndex * n + s];
            }

            // If this was larger, then overwrite piPrime and GammaPrime's values.
            if (actionValue > maxActionValue) {
                maxActionValue = actionValue;

                piPrime[beliefIndex] = action;
            }
        }

        __syncthreads();
    }

    for (unsigned int s = 0; s < n; s++) {
        GammaPrime[beliefIndex * n + s] = alphaBA[beliefIndex * m * n +
                                                piPrime[beliefIndex] * n + s];
    }
}


int pomdp_pbvi_complete_gpu(unsigned int n, unsigned int m, unsigned int z, unsigned int r,
        unsigned int maxNonZeroBeliefs, unsigned int maxSuccessors,
        const float *B, const float *T, const float *O, const float *R,
        const bool *available, const int *nonZeroBeliefs, const int *successors,
        float gamma, unsigned int horizon, unsigned int numThreads,
        float *Gamma, unsigned int *pi)
{
    float *d_B;
    float *d_T;
    float *d_O;
    float *d_R;
    bool *d_available;
    int *d_nonZeroBeliefs;
    int *d_successors;

    int result;

    result = pomdp_initialize_belief_points_gpu(n, r, B, d_B);
    if (result != NOVA_SUCCESS) {
        return result;
    }
    result = pomdp_initialize_state_transitions_gpu(n, m, T, d_T);
    if (result != NOVA_SUCCESS) {
        return result;
    }
    result = pomdp_initialize_observation_transitions_gpu(n, m, z, O, d_O);
    if (result != NOVA_SUCCESS) {
        return result;
    }
    result = pomdp_initialize_rewards_gpu(n, m, R, d_R);
    if (result != NOVA_SUCCESS) {
        return result;
    }
    result = pomdp_initialize_available_gpu(m, r, available, d_available);
    if (result != NOVA_SUCCESS) {
        return result;
    }
    result = pomdp_initialize_nonzero_beliefs_gpu(r, maxNonZeroBeliefs, nonZeroBeliefs, d_nonZeroBeliefs);
    if (result != NOVA_SUCCESS) {
        return result;
    }
    result = pomdp_initialize_successors_gpu(n, m, maxSuccessors, successors, d_successors);
    if (result != NOVA_SUCCESS) {
        return result;
    }

    result = pomdp_pbvi_execute_gpu(n, m, z, r, maxNonZeroBeliefs, maxSuccessors,
            d_B, d_T, d_O, d_R, d_available, d_nonZeroBeliefs, d_successors,
            gamma, horizon, numThreads, Gamma, pi);
    if (result != NOVA_SUCCESS) {
        return result;
    }

    result = NOVA_SUCCESS;
    if (pomdp_uninitialize_belief_points_gpu(d_B) != NOVA_SUCCESS) {
        result = NOVA_ERROR_DEVICE_FREE;
    }
    if (pomdp_uninitialize_state_transitions_gpu(d_T) != NOVA_SUCCESS) {
        result = NOVA_ERROR_DEVICE_FREE;
    }
    if (pomdp_uninitialize_observation_transitions_gpu(d_O) != NOVA_SUCCESS) {
        result = NOVA_ERROR_DEVICE_FREE;
    }
    if (pomdp_uninitialize_rewards_gpu(d_R) != NOVA_SUCCESS) {
        result = NOVA_ERROR_DEVICE_FREE;
    }
    if (pomdp_uninitialize_available_gpu(d_available) != NOVA_SUCCESS) {
        result = NOVA_ERROR_DEVICE_FREE;
    }
    if (pomdp_uninitialize_nonzero_beliefs_gpu(d_nonZeroBeliefs) != NOVA_SUCCESS) {
        result = NOVA_ERROR_DEVICE_FREE;
    }
    if (pomdp_uninitialize_successors_gpu(d_successors) != NOVA_SUCCESS) {
        result = NOVA_ERROR_DEVICE_FREE;
    }

    return result;
}


int pomdp_pbvi_initialize_gpu(unsigned int n, unsigned int m, unsigned int r,
        unsigned int numThreads, float *Gamma, float *&d_Gamma, float *&d_GammaPrime,
        unsigned int *&d_pi, unsigned int *&d_piPrime, float *&d_alphaBA,
        unsigned int *numBlocks)
{
    *numBlocks = (unsigned int)((float)r / (float)numThreads) + 1;

    // Create the device-side Gamma.
    if (cudaMalloc(&d_Gamma, r * n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_initialize_gpu]: %s",
                "Failed to allocate device-side memory for Gamma.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }
    if (cudaMemcpy(d_Gamma, Gamma, r * n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_initialize_gpu]: %s",
                "Failed to copy memory from host to device for Gamma.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    if (cudaMalloc(&d_GammaPrime, r * n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_initialize_gpu]: %s",
                "Failed to allocate device-side memory for Gamma (prime).");
        return NOVA_ERROR_DEVICE_MALLOC;
    }
    if (cudaMemcpy(d_GammaPrime, Gamma, r * n * sizeof(float),
                    cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_initialize_gpu]: %s",
                "Failed to copy memory from host to device for Gamma (prime).");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    // Create the device-side pi.
    if (cudaMalloc(&d_pi, r * sizeof(unsigned int)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_initialize_gpu]: %s",
                "Failed to allocate device-side memory for pi.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }
    if (cudaMalloc(&d_piPrime, r * sizeof(unsigned int)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_initialize_gpu]: %s",
                "Failed to allocate device-side memory for pi (prime).");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    // Create the device-side memory for the intermediate variable alphaBA.
    if (cudaMalloc(&d_alphaBA, r * m * n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_initialize_gpu]: %s",
                "Failed to allocate device-side memory for alphaBA.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    return NOVA_SUCCESS;
}


int pomdp_pbvi_update_gpu(unsigned int n, unsigned int m, unsigned int z, unsigned int r,
        unsigned int maxNonZeroBeliefs, unsigned int maxSuccessors,
        const float *d_B, const float *d_T, const float *d_O, const float *d_R,
        const bool *d_available, const int *d_nonZeroBeliefs, const int *d_successors,
        float gamma, unsigned int currentHorizon, unsigned int numThreads, unsigned int numBlocks,
        float *d_Gamma, float *d_GammaPrime, unsigned int *d_pi, unsigned int *d_piPrime,
        float *d_alphaBA)
{
    pomdp_pbvi_initialize_alphaBA_gpu<<< dim3(r, m, 1), numThreads >>>(n, m, r, d_R, d_alphaBA);

    // Check if there was an error executing the kernel.
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_update_gpu]: %s",
                        "Failed to execute the 'initialization of alphaBA' kernel.");
        return NOVA_ERROR_KERNEL_EXECUTION;
    }

    // Wait for the kernel to finish before looping more.
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_update_gpu]: %s",
                    "Failed to synchronize the device after 'initialization of alphaBA' kernel.");
        return NOVA_ERROR_DEVICE_SYNCHRONIZE;
    }

    pomdp_pbvi_compute_alphaBA_gpu<<< dim3(r, m, z), numThreads,
                numThreads * sizeof(float) + numThreads * sizeof(unsigned int) >>>(
                n, m, z, r,
                maxNonZeroBeliefs, maxSuccessors,
                d_B, d_T, d_O, d_R,
                d_available, d_nonZeroBeliefs, d_successors,
                gamma,
                d_Gamma, d_pi,
                d_alphaBA);

    // Check if there was an error executing the kernel.
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_update_gpu]: %s",
                        "Failed to execute the 'compute_alphaBA' kernel.");
        return NOVA_ERROR_KERNEL_EXECUTION;
    }

    // Wait for the kernel to finish before looping more.
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_update_gpu]: %s",
                        "Failed to synchronize the device after 'compute_alphaBA' kernel.");
        return NOVA_ERROR_DEVICE_SYNCHRONIZE;
    }

    // Execute a kernel for the first three stages of for-loops: B, A, Z, as a 3d-block,
    // and the 4th stage for-loop over Gamma as the threads.
    if (currentHorizon % 2 == 0) {
        pomdp_pbvi_update_step_gpu<<< numBlocks, numThreads >>>(n, m, z, r,
                maxNonZeroBeliefs, maxSuccessors,
                d_B, d_T, d_O, d_R,
                d_available, d_nonZeroBeliefs, d_successors,
                gamma,
                d_Gamma, d_pi,
                d_alphaBA,
                d_GammaPrime, d_piPrime);
    } else {
        pomdp_pbvi_update_step_gpu<<< numBlocks, numThreads >>>(n, m, z, r,
                maxNonZeroBeliefs, maxSuccessors,
                d_B, d_T, d_O, d_R,
                d_available, d_nonZeroBeliefs, d_successors,
                gamma,
                d_GammaPrime, d_piPrime,
                d_alphaBA,
                d_Gamma, d_pi);
    }

    // Check if there was an error executing the kernel.
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_update_gpu]: %s",
                        "Failed to execute the 'pomdp_pbvi_update_step_gpu' kernel.");
        return NOVA_ERROR_KERNEL_EXECUTION;
    }

    // Wait for the kernel to finish before looping more.
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_pbvi_update_gpu]: %s",
                        "Failed to synchronize the device after 'pomdp_pbvi_update_step_gpu' kernel.");
        return NOVA_ERROR_DEVICE_SYNCHRONIZE;
    }

    return NOVA_SUCCESS;
}


int pomdp_pbvi_get_policy_gpu(unsigned int n, unsigned int r, unsigned int horizon,
        const float *d_Gamma, const float *d_GammaPrime, const unsigned int *d_pi,
        const unsigned int *d_piPrime, float *Gamma, unsigned int *pi)
{
    // Copy the final result of Gamma and pi to the variables. This assumes
    // that the memory has been allocated.
    if (horizon % 2 == 1) {
        if (cudaMemcpy(Gamma, d_Gamma, r * n * sizeof(float),
                        cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_pbvi_get_policy_gpu]: %s",
                    "Failed to copy memory from device to host for Gamma.");
            return NOVA_ERROR_MEMCPY_TO_HOST;
        }
        if (cudaMemcpy(pi, d_pi, r * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_pbvi_get_policy_gpu]: %s",
                    "Failed to copy memory from device to host for pi.");
            return NOVA_ERROR_MEMCPY_TO_HOST;
        }
    } else {
        if (cudaMemcpy(Gamma, d_GammaPrime, r * n * sizeof(float),
                        cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_pbvi_get_policy_gpu]: %s",
                    "Failed to copy memory from device to host for Gamma (prime).");
            return NOVA_ERROR_MEMCPY_TO_HOST;
        }
        if (cudaMemcpy(pi, d_piPrime, r * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_pbvi_get_policy_gpu]: %s",
                    "Failed to copy memory from device to host for pi (prime).");
            return NOVA_ERROR_MEMCPY_TO_HOST;
        }
    }

    return NOVA_SUCCESS;
}


int pomdp_pbvi_uninitialize_gpu(float *&d_Gamma, float *&d_GammaPrime,
        unsigned int *&d_pi, unsigned int *&d_piPrime, float *&d_alphaBA)
{
    if (d_Gamma != nullptr) {
        if (cudaFree(d_Gamma) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_pbvi_uninitialize_gpu]: %s",
                    "Failed to allocate device-side memory for the Gamma (the alpha-vectors).");
        }
    }
    d_Gamma = nullptr;

    if (d_GammaPrime != nullptr) {
        if (cudaFree(d_GammaPrime) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_pbvi_uninitialize_gpu]: %s",
                    "Failed to allocate device-side memory for the GammaPrime (the alpha-vectors' copy).");
        }
    }
    d_GammaPrime = nullptr;

    if (d_pi != nullptr) {
        if (cudaFree(d_pi) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_pbvi_uninitialize_gpu]: %s",
                    "Failed to allocate device-side memory for the pi (the policy).");
        }
    }
    d_pi = nullptr;

    if (d_piPrime != nullptr) {
        if (cudaFree(d_piPrime) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_pbvi_uninitialize_gpu]: %s",
                    "Failed to allocate device-side memory for the piPrime (the policy copy).");
        }
    }
    d_piPrime = nullptr;

    if (d_alphaBA != nullptr) {
        if (cudaFree(d_alphaBA) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_pbvi_uninitialize_gpu]: %s",
                    "Failed to allocate device-side memory for alphaBA (alpha-vector collection).");
        }
    }
    d_alphaBA = nullptr;

    return NOVA_SUCCESS;
}


int pomdp_pbvi_execute_gpu(unsigned int n, unsigned int m, unsigned int z, unsigned int r,
        unsigned int maxNonZeroBeliefs, unsigned int maxSuccessors,
        const float *d_B, const float *d_T, const float *d_O, const float *d_R,
        const bool *d_available, const int *d_nonZeroBeliefs, const int *d_successors,
        float gamma, unsigned int horizon, unsigned int numThreads,
        float *Gamma, unsigned int *pi)
{
    // The device pointers for the alpha-vectors: Gamma and GammaPrime.
    float *d_Gamma;
    float *d_GammaPrime;

    // The device pointers for the actions taken on each alpha-vector: pi and piPrime.
    unsigned int *d_pi;
    unsigned int *d_piPrime;

    // The device pointer for the intermediate alpha-vectors computed in the inner for loop.
    float *d_alphaBA;

    // The number of blocks to execute in the main PBVI step.
    unsigned int numBlocks;

    // The result from calling other functions.
    int result;

    // Ensure the data is valid.
    if (n == 0 || m == 0 || z == 0 || r == 0 || maxNonZeroBeliefs == 0 || maxSuccessors == 0 ||
            d_B == nullptr || d_T == nullptr || d_O == nullptr || d_R == nullptr ||
            d_available == nullptr || d_nonZeroBeliefs == nullptr || d_successors == nullptr ||
            gamma < 0.0 || gamma >= 1.0 || horizon < 1) {
        fprintf(stderr, "Error[pomdp_pbvi_execute_gpu]: %s", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Ensure threads are correct.
    if (numThreads % 32 != 0) {
        fprintf(stderr, "Error[pomdp_pbvi_execute_gpu]: %s", "Invalid number of threads.");
        return NOVA_ERROR_INVALID_CUDA_PARAM;
    }

    result = pomdp_pbvi_initialize_gpu(n, m, r, numThreads, Gamma, d_Gamma, d_GammaPrime,
                                d_pi, d_piPrime, d_alphaBA, &numBlocks);
    if (result != NOVA_SUCCESS) {
        return result;
    }

    // For each of the updates, run PBVI.
    for (int t = 0; t < horizon; t++) {
        result = pomdp_pbvi_update_gpu(n, m, z, r, maxNonZeroBeliefs, maxSuccessors,
                d_B, d_T, d_O, d_R,
                d_available, d_nonZeroBeliefs, d_successors,
                gamma, t, numThreads, numBlocks,
                d_Gamma, d_GammaPrime, d_pi, d_piPrime, d_alphaBA);
        if (result != NOVA_SUCCESS) {
            return result;
        }
    }

    result = pomdp_pbvi_get_policy_gpu(n, r, horizon, d_Gamma, d_GammaPrime, d_pi, d_piPrime, Gamma, pi);
    if (result != NOVA_SUCCESS) {
        return result;
    }

    result = pomdp_pbvi_uninitialize_gpu(d_Gamma, d_GammaPrime, d_pi, d_piPrime, d_alphaBA);
    if (result != NOVA_SUCCESS) {
        return result;
    }

    return NOVA_SUCCESS;
}


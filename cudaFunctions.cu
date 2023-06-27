#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

__device__ double calcDistance(const Point *p1, const Point *p2, double *t)
{
    double x1 = ((p1->x2 - p1->x1) / 2) * sin((*t) * M_PI / 2) + ((p1->x2 + p1->x1) / 2);
    double y1 = p1->a * x1 + p1->b;

    double x2 = ((p2->x2 - p2->x1) / 2) * sin((*t) * M_PI / 2) + ((p2->x2 + p2->x1) / 2);
    double y2 = p2->a * x2 + p2->b;

    double distance = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));

    return distance;
}

__global__ void checkProximityCriteria(int *count, Point *points, double *tValues, const int tCount, const int N, const int K, const double D, int *results)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // point idx
    if (idx < tCount)
    {
        double t = tValues[idx];

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N && i != j; j++)
            {
                double distance = calcDistance(&points[i], &points[j], &t);

                if (distance <= D && distance > 0)
                {
                    int currentCount = atomicAdd(count, 1);
                    if (currentCount < K)
                    {
                        int index = idx * tCount + i;
                        printf("t = %d || at index_res = %d save point %d\n",idx, index, points[i].id);
                        atomicExch(&results[index], points[i].id);

                        // Check if all K results have been found
                        if (*count >= K)
                            return;
                    }
                }
            }
        }
        // __threadfence();
        // __syncthreads();
        for (int i = 0; i < tCount; i++)
        {
            printf("current t %d\n", i);
            for (int j = 0; j < N; j++)
            {
                printf("\tp[%d] = %d ", j, results[i * N + j]);
            }
            printf("\n");
        }
    }
}

void computeOnGPU(int *count, int *N, int *K, double *D, int *tCountSize, double *myTValues, Point *points, int *results)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    int threadPerBlock = *tCountSize < BLOCK_SIZE ? *tCountSize : BLOCK_SIZE;
    int blocksPerGrid = ((*tCountSize) * (*N)) / threadPerBlock < 1 ? 1 : ceil(((*tCountSize) * (*N)) / threadPerBlock);
    int *d_count = NULL;
    Point *d_points = NULL;
    double *d_tValues = NULL;
    int *d_results = NULL;

    // Allocate the device
    err = cudaMalloc((void **)&d_count, sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device count (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_points, (*N) * sizeof(Point));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device points (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_tValues, (*tCountSize) * sizeof(double));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device tValues (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_results, (*N) * (*tCountSize) * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device results (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy from host to device
    err = cudaMemcpy(d_points, points, (*N) * sizeof(Point), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy points from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_tValues, myTValues, (*tCountSize) * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy tValues from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_count, count, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy count from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_results, results, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy results from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the proximity criteria check on the GPU
    checkProximityCriteria<<<blocksPerGrid, threadPerBlock>>>(d_count, d_points, d_tValues, *tCountSize, *N, *K, *D, d_results);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch checkProximityCriteria kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy from device to host
    err = cudaMemcpy(count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy count from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(results, d_results, (*N) * (*tCountSize) * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy results from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device memory
    cudaFree(d_count);
    cudaFree(d_points);
    cudaFree(d_tValues);
    cudaFree(d_results);
}

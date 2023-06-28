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

__device__ bool isProximityCriteriaMet(const Point *p1, const Point *p2, double *t, double D)
{
    double distance = calcDistance(p1, p2, t);
    // printf("t = %.3lf || point %d, point %d || distance = %.3lf\n",*t,p1->id,p2->id,distance);
    return distance <= D;
}

__device__ void updateResults(int idx, int *results, int proximityPointId)
{
    for (int j = 0; j < CONSTRAINTS; j++)
    {
        if (results[idx * CONSTRAINTS + j] == -1)
        {
            atomicExch(&results[idx * CONSTRAINTS + j], proximityPointId);
            // printf("res_p%d || idx_t = %d || with point %d || res_index = %d\n",
            // j,idx,proximityPointId, idx * CONSTRAINTS + j);
            break;
        }
    }
}

__global__ void checkProximityCriteria(Point *points, double *tValues, const int tCount, const int N, const int K, const double D, int *results)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // point idx

    if (idx >= tCount) return; // specific t

    double t = tValues[idx];
    // printf("t = %d value = %.3lf\n",idx, t);
    int count = 0;
    int finish = 0;

    for (int i = 0; i < N; i++)
    {
        count = 0;
        finish = 0;
        for (int j = 0; j < N; j++)
        {
            if (finish == 1) break;
            if (i != j && isProximityCriteriaMet(&points[i], &points[j], &t, D))
            {
                count++;
                if (count == K)
                {
                    int proximityPointId = points[i].id;
                    updateResults(idx, results, proximityPointId);
                    finish = 1;
                }
            }
        }
    }
}

void computeOnGPU(int *N, int *K, double *D, int *tCountSize, double *myTValues, Point *points, int *results)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // TODO: need to fix this section !!!/////////////////////////////
    int threadPerBlock = min(BLOCK_SIZE, *tCountSize);              //
    int blocksPerGrid = *tCountSize / BLOCK_SIZE < 1 ? 1 : *tCountSize / BLOCK_SIZE; //
    // printf("*tCountSize = %d threadPerBlock=%d blocksPerGrid=%d\n", //
    //  *tCountSize,threadPerBlock,blocksPerGrid);                      //
    //////////////////////////////////////////////////////////////////

    Point *d_points = NULL;
    double *d_tValues = NULL;
    int *d_results = NULL;

    // Allocate the device
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
    err = cudaMalloc((void **)&d_results, (CONSTRAINTS) * (*tCountSize) * sizeof(int));
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
    err = cudaMemcpy(d_results, results, (CONSTRAINTS) * (*tCountSize) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy results from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the proximity criteria check on the GPU
    checkProximityCriteria<<<blocksPerGrid, threadPerBlock>>>(d_points, d_tValues, *tCountSize, *N, *K, *D, d_results);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch checkProximityCriteria kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy from device to host
    err = cudaMemcpy(results, d_results, (CONSTRAINTS) * (*tCountSize) * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy results from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < *tCountSize; i++)
    {
        printf("current t %d\n", i);
        for (int j = 0; j < CONSTRAINTS; j++)
        {
            printf("\tp[%d] = %d ", j, results[i * (CONSTRAINTS) + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_points);
    cudaFree(d_tValues);
    cudaFree(d_results);
}

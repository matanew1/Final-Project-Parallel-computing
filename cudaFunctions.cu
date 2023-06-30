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
    return distance <= D;
}

__device__ void updateResults(int idx, int *results, int proximityPointId)
{
    for (int j = 0; j < CONSTRAINTS; j++)
    {
        int targetIndex = idx * CONSTRAINTS + j;
        if (results[targetIndex] == -1)
        {
            atomicExch(&results[targetIndex], proximityPointId);
            return;
        }
    }
}

__global__ void checkProximityCriteria(Point *points, double *tValues, const int tCount, const int N, const int K, const double D, int *results)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // point idx

    if (idx >= tCount)
        return; // specific t

    double t = tValues[idx];
    int count = 0;

    for (int i = 0; i < N; i++)
    {
        count = 0;
        for (int j = 0; j < N; j++)
        {
            if (i != j && isProximityCriteriaMet(&points[i], &points[j], &t, D))
            {
                count++;
                if (count == K)
                {
                    int proximityPointId = points[i].id;
                    updateResults(idx, results, proximityPointId);
                    break;
                }
            }
        }
    }
}

void allocateDeviceMemory(void **devPtr, size_t size)
{
    cudaError_t err = cudaMalloc(devPtr, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void copyHostToDevice(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    cudaError_t err = cudaMemcpy(dst, src, count, kind);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void copyDeviceToHost(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    cudaError_t err = cudaMemcpy(dst, src, count, kind);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void freeDeviceMemory(void *devPtr)
{
    cudaFree(devPtr);
}

void computeOnGPU(int *N, int *K, double *D, int *tCountSize, double *myTValues, Point *points, int *results)
{
    cudaError_t err = cudaSuccess;
    int threadPerBlock = min(BLOCK_SIZE, *tCountSize);
    int blocksPerGrid = (*tCountSize + threadPerBlock - 1) / threadPerBlock;

    Point *d_points = NULL;
    double *d_tValues = NULL;
    int *d_results = NULL;

    allocateDeviceMemory((void **)&d_points, (*N) * sizeof(Point));
    allocateDeviceMemory((void **)&d_tValues, (*tCountSize) * sizeof(double));
    allocateDeviceMemory((void **)&d_results, CONSTRAINTS * (*tCountSize) * sizeof(int));

    copyHostToDevice(d_points, points, (*N) * sizeof(Point), cudaMemcpyHostToDevice);
    copyHostToDevice(d_tValues, myTValues, (*tCountSize) * sizeof(double), cudaMemcpyHostToDevice);
    copyHostToDevice(d_results, results, CONSTRAINTS * (*tCountSize) * sizeof(int), cudaMemcpyHostToDevice);

    checkProximityCriteria<<<blocksPerGrid, threadPerBlock>>>(d_points, d_tValues, *tCountSize, *N, *K, *D, d_results);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch checkProximityCriteria kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    copyDeviceToHost(results, d_results, CONSTRAINTS * (*tCountSize) * sizeof(int), cudaMemcpyDeviceToHost);

    freeDeviceMemory(d_points);
    freeDeviceMemory(d_tValues);
    freeDeviceMemory(d_results);
}

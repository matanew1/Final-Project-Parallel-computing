#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

/**
 * Calculate the distance between two points at a given t value.
 *
 * @param p1  Pointer to the first point
 * @param p2  Pointer to the second point
 * @param t   Pointer to the t value
 * @return    The calculated distance
 */
__device__ double calcDistance(const Point *p1, const Point *p2, double *t)
{
    double x1 = ((p1->x2 - p1->x1) / 2) * sin((*t) * M_PI / 2) + ((p1->x2 + p1->x1) / 2);
    double y1 = p1->a * x1 + p1->b;

    double x2 = ((p2->x2 - p2->x1) / 2) * sin((*t) * M_PI / 2) + ((p2->x2 + p2->x1) / 2);
    double y2 = p2->a * x2 + p2->b;

    double distance = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));

    return distance;
}

/**
 * Check if the proximity criteria is met between two points at a given t value.
 *
 * @param p1  Pointer to the first point
 * @param p2  Pointer to the second point
 * @param t   Pointer to the t value
 * @param D   Proximity criteria threshold
 * @return    True if the criteria is met, false otherwise
 */
__device__ bool isProximityCriteriaMet(const Point *p1, const Point *p2, double *t, double D)
{
    double distance = calcDistance(p1, p2, t);
    return distance <= D;
}

/**
 * Update the results array with the proximity point ID.
 *
 * @param idx              Index of the results array
 * @param results          Pointer to the results array
 * @param proximityPointId Proximity point ID
 */
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

/**
 * GPU kernel function to check the proximity criteria for points and update the results array.
 *
 * @param points     Array of points
 * @param tValues    Array of t values
 * @param tCount     Total number of t values
 * @param N          Number of points
 * @param K          Number of points to satisfy proximity criteria
 * @param D          Proximity criteria threshold
 * @param results    Results array
 */
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

/**
 * Allocate device memory.
 *
 * @param devPtr  Pointer to the allocated device memory
 * @param size    Size of the memory to allocate
 */
void allocateDeviceMemory(void **devPtr, size_t size)
{
    cudaError_t err = cudaMalloc(devPtr, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
 * Copy data from host to device.
 *
 * @param dst    Pointer to the destination in device memory
 * @param src    Pointer to the source in host memory
 * @param count  Number of bytes to copy
 * @param kind   Type of cudaMemcpy operation
 */
void copyHostToDevice(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    cudaError_t err = cudaMemcpy(dst, src, count, kind);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
 * Copy data from device to host.
 *
 * @param dst    Pointer to the destination in host memory
 * @param src    Pointer to the source in device memory
 * @param count  Number of bytes to copy
 * @param kind   Type of cudaMemcpy operation
 */
void copyDeviceToHost(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    cudaError_t err = cudaMemcpy(dst, src, count, kind);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
 * Free device memory.
 *
 * @param devPtr  Pointer to the device memory to free
 */
void freeDeviceMemory(void *devPtr)
{
    cudaFree(devPtr);
}

/**
 * Compute the proximity criteria on the GPU.
 *
 * @param N            Number of points
 * @param K            Number of points to satisfy proximity criteria
 * @param D            Proximity criteria threshold
 * @param tCountSize   Size of the t values for each process
 * @param myTValues    T values for each process
 * @param points       Array of points
 * @param results      Results array
 */
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

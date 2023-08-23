#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

/**
 * @brief this function will calc the distance between two points
 * @param p1 Struct that points on one point.
 * @param p2 Struct that points on the secound point.
 * @param t The current tvalue
 */
__device__ double calcDistance(const Point p1, const Point p2, double t)
{
    double x1 = ((p1.x2 - p1.x1) / 2) * __sinf(t * M_PI / 2) + ((p1.x2 + p1.x1) / 2);
    double y1 = p1.a * x1 + p1.b;

    double x2 = ((p2.x2 - p2.x1) / 2) * __sinf(t * M_PI / 2) + (p2.x2 + p2.x1) / 2;
    double y2 = p2.a * x2 + p2.b;

    double dx = x2 - x1;
    double dy = y2 - y1;

    return sqrt(dx * dx + dy * dy);
}
/**
 * @brief In this function points that have at least K points that the distance less then D will enter to the function
 * and update the array of results.
 * @param startingIndex the current tvalue in the round
 * @param resutls array of results
 * @param pointId A point that satisfies the condition
 */
__device__ void updateProximitePoints(int startingIndex, int *resutls, int pointId)
{
    for (int i = 0; i < CONSTRAINTS; i++)
    {
        int index = startingIndex * CONSTRAINTS + i;
        int currentVal = resutls[index];
        if (currentVal == -1)
        {
            int expected = -1;
            int desired = pointId;

            if (atomicCAS(&resutls[index], expected, desired) == expected) /*Fill the resutls[index] in atomic way*/
            {
                return;
            }
        }
    }
}
/**
 * @brief this function will calcluate for each Tvalue[tIndex] Checks if it exists a Proximity Criteria.
 * Each thread will get point and check with other points if there is another point if the distance is smaller then D.
 * but before checking it, we will check the last poistion of d_resutls[tIndex * CONSTRAINTS + CONSTRAINTS - 1].
 * explanation: All CONSTRAINTS cells represent K points that Proximity Criteria. if there at Least K points fulfill the condition the points will
 * enter to the d_resutls.
 * Note: if one of the threads fill the last point in  d_resutls[tIndex * CONSTRAINTS + CONSTRAINTS - 1] the other threads will not continue
 * calclute the distance.
 * @param d_points array of all N points
 * @param N numbers of points
 * @param tValue current tValue
 * @param D max distance to check
 * @param d_resutls array of result
 * @param K need at least K points that fulfill the condition of Poximity Criteria
 * @param tIndex the current t in the for loop


*/
__global__ void calculateProximity(Point *d_points, int N, double tValue, double D, int *d_resutls, double K, int tIndex)
{
    int pid = blockDim.x * blockIdx.x + threadIdx.x;
    int counter = 0;
    if (pid < N && atomicAdd(&d_resutls[tIndex * CONSTRAINTS + CONSTRAINTS - 1], 0) == -1)
    {
        for (int i = 0; i < N; i++)
        {

            if (atomicAdd(&d_resutls[tIndex * CONSTRAINTS + CONSTRAINTS - 1], 0) != -1)
            {
                return;
            }

            if (d_points[i].id != d_points[pid].id && calcDistance(d_points[pid], d_points[i], tValue) < D)
            {
                counter++;
                if (counter == K)
                {

                    int pointId = d_points[pid].id;                        /*This point is proximity*/
                    updateProximitePoints(tIndex, d_resutls, pointId); /*0,resutls,point that have 3 points*/
                    break;
                }
            }
        }
    }
}

void allocateMemDevice(void **ptr, size_t size)
{
    cudaError_t err = cudaMalloc(ptr, size); /*Allocate mem on device*/
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Cannot to allocate memory on device. -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
/**
 * @param dest Pointer to the dest in device / host memory
 * @param src  Pointer to the src in host / device memory
 * @param size How much data need to copy(Bytes)
 * @param direction Which direction to copy the mem(host-> device) | (device -> host)
 */
void copyMemory(void *dest, void *src, size_t size, cudaMemcpyKind direction)
{
    /*Copy mem from device to host OR host to device depending on the direction*/
    cudaError_t err = cudaMemcpy(dest, src, size, direction);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data. -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
// 0528626957 boris
int computeOnGPU(int N, int K, double D, int tCount, double *tValues, Point *points, int *results)
{
    cudaError_t err = cudaSuccess;

    int threadPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadPerBlock - 1) / threadPerBlock;
    Point *d_points;
    int *d_resutls = NULL;
    /*Allocating mem on gpu section*/
    allocateMemDevice((void **)&d_resutls, CONSTRAINTS * tCount * sizeof(int)); // resutls == globalflags
    allocateMemDevice((void **)&d_points, N * sizeof(Point));
    /*End allocate mem*/

    /*Copy mem to Device section*/
    copyMemory(d_points, points, N * sizeof(Point), cudaMemcpyHostToDevice);
    copyMemory(d_resutls, results, tCount * CONSTRAINTS * sizeof(int), cudaMemcpyHostToDevice);
    /*End copy mem to Device*/

    /*for each tvalue we will send it to GPU to compute the data and save it on resutls array*/
    for (int i = 0; i < tCount; i++)
    {
        calculateProximity<<<blocksPerGrid, threadPerBlock>>>(d_points, N, tValues[i], D, d_resutls, K, i);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to lanch calculateProximity kernel. -%s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    err = cudaMemcpy(results, d_resutls, tCount * CONSTRAINTS * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data. -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data. -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if (cudaFree(d_points) != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    if (cudaFree(d_resutls) != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return 0;
}

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

__device__ int satisfyProximityCriteria(const Point* p1, const Point* p2, const Point* p3, double* t, int *K, double *D) {
    double x1 = ((p1->x2 - p1->x1) / 2) * sin(*t * M_PI / 2) + (p1->x2 + p1->x1) / 2;
    double x2 = ((p2->x2 - p2->x1) / 2) * sin(*t * M_PI / 2) + (p2->x2 + p2->x1) / 2;
    double x3 = ((p3->x2 - p3->x1) / 2) * sin(*t * M_PI / 2) + (p3->x2 + p3->x1) / 2;

    double y1 = p1->a * x1 + p1->b;
    double y2 = p2->a * x2 + p2->b;
    double y3 = p3->a * x3 + p3->b;

    double dist12 = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
    double dist13 = sqrt(pow(x3 - x1, 2) + pow(y3 - y1, 2));
    double dist23 = sqrt(pow(x3 - x2, 2) + pow(y3 - y2, 2));

    return (dist12 < *D && dist13 < *D && dist23 < *D);
}

__global__ void checkProximityCriteria(const Point *points, double *tValues, int *N, int *K, double *D,int *results, int *resultsCount){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // point idx
    printf("idx = %d\n", idx);
    if (idx < *N)
    {
        double t = tValues[blockIdx.y];
        // printf("[%d] t=[%d]\n",idx,t);
        // for (int j = idx + 1; j < *N - 1; ++j)
        // {
        //     for (int k = j + 1; k < *N; ++k)
        //     {
        //         // if (satisfyProximityCriteria(&points[idx], &points[j], &points[k], &t, K, D)) {
        //         //     int resIdx = atomicAdd(resultsCount, 1);
        //         //     results[resIdx * 3] = points[idx].id;
        //         //     results[resIdx * 3 + 1] = points[j].id;
        //         //     results[resIdx * 3 + 2] = points[k].id;
        //         // }
        //     }
        // }
    }
}

/**
 * N - number of points 
 * D - radius around the current point
 * K - number of points until distance of D
 * tCount - number of points (t), we want to check
 */
int computeOnGPU(int *N, int *K, double *D, int *tCount, int *myPointsCount, double *tValues, int *results, int *resultsCount, int *maxResults, Point *myPoints)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    int blocksPerGrid = (*myPointsCount + BLOCK_SIZE - 1) / BLOCK_SIZE; 
    int threadsPerBlock = BLOCK_SIZE;

    Point *dPoints;     // point for device
    double *dTValues;   // tValues for device
    int *dResults;      // results for device
    int *dResultsCount; // resultsCount for device

    err = cudaMalloc(&dPoints, (*myPointsCount) * sizeof(Point));
    if (err != cudaSuccess){
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc(&dTValues, (*tCount + 1) * sizeof(double));
    if (err != cudaSuccess){
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc(&dResults, (*maxResults) * 3 * sizeof(int));
    if (err != cudaSuccess){
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc(&dResultsCount, sizeof(int));
    if (err != cudaSuccess){
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(dPoints, myPoints, (*myPointsCount) * sizeof(Point), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(dTValues, tValues, (*tCount + 1) * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    checkProximityCriteria<<<blocksPerGrid, threadsPerBlock>>>(dPoints, dTValues, N, K, D, dResults, dResultsCount);

    // err = cudaMemcpy(&resultsCount, dResultsCount, sizeof(int), cudaMemcpyDeviceToHost);
    // if (err != cudaSuccess){
    //     fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // err = cudaMemcpy(results, dResults, (*resultsCount) * 3 * sizeof(int), cudaMemcpyDeviceToHost);
    // if (err != cudaSuccess){
    //     fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // free device allocation
    cudaFree(dPoints);
    cudaFree(dTValues);
    cudaFree(dResults);
    cudaFree(dResultsCount);
    return 0;
}

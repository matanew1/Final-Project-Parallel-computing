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

__global__ void checkProximityCriteria(const Point *points, const double *tValues, const int tCount,const int N,const int K, const double D){
int idx = blockIdx.x * blockDim.x + threadIdx.x; // point idx
    if (idx < tCount)
    {
        const int currentTValue = tValues[idx];
        int count = 0;
        for (int i = 0; i < N; i++) {
            const Point currentPoint = points[i];
            double x = ((currentPoint.x2- currentPoint.x1) / 2) * sin(currentTValue * M_PI / 2) + ((currentPoint.x2 + currentPoint.x1) / 2);
            double y = currentPoint.a * x + currentPoint.b;
            printf("Current t = %d  i = %d) point ---> x= %lf y= %lf\n",idx,i,x,y);
                // if (distance < D) {
                //     count++;
                //     if (count >= K) {
                //         break;
                //     }
                // }            
        }
    }
}

/**
 * N - number of points 
 * D - radius around the current point
 * K - number of points until distance of D
 * tCount - number of points (t), we want to check
 */
void computeOnGPU(int *N, int *K, double *D, int *tCountSize, double *myTValues, Point *points) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    int blocksPerGrid = ((*tCountSize) + BLOCK_SIZE - 1) / BLOCK_SIZE; 
    int threadsPerBlock = BLOCK_SIZE;

    Point *dPoints;     // point for device
    double *dTValues;   // tValues for device    

    err = cudaMalloc(&dPoints, (*N) * sizeof(Point));
    if (err != cudaSuccess){
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc(&dTValues, (*tCountSize) * sizeof(double));
    if (err != cudaSuccess){
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(dPoints, points, (*N) * sizeof(Point), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(dTValues, myTValues, (*tCountSize) * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    checkProximityCriteria<<<blocksPerGrid, threadsPerBlock>>>(dPoints, dTValues, *tCountSize, *N, *K, *D);

    // free device allocation
    cudaFree(dPoints);
    cudaFree(dTValues);
}

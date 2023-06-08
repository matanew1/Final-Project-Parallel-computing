#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

// __global__ void checkProximityCriteria(const struct Point* points, double* tValues, int N, int K, double D, int* results, int* resultsCount) {
//     double x1 = ((p1->x2 - p1->x1) / 2) * sin(t * M_PI / 2) + (p1->x2 + p1->x1) / 2;
//     double x2 = ((p2->x2 - p2->x1) / 2) * sin(t * M_PI / 2) + (p2->x2 + p2->x1) / 2;
//     double x3 = ((p3->x2 - p3->x1) / 2) * sin(t * M_PI / 2) + (p3->x2 + p3->x1) / 2;

//     double y1 = p1->a * x1 + p1->b;
//     double y2 = p2->a * x2 + p2->b;
//     double y3 = p3->a * x3 + p3->b;

//     double dist12 = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
//     double dist13 = sqrt(pow(x3 - x1, 2) + pow(y3 - y1, 2));
//     double dist23 = sqrt(pow(x3 - x2, 2) + pow(y3 - y2, 2));

//     return (dist12 < D && dist13 < D && dist23 < D);
// }
// __device__ int satisfyProximityCriteria(const struct Point* p1, const struct Point* p2, const struct Point* p3, double t, int K, double D) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < N) {
//         double t = tValues[blockIdx.y];

//         for (int j = idx + 1; j < N - 1; ++j) {
//             for (int k = j + 1; k < N; ++k) {
//                 if (satisfyProximityCriteria(&points[idx], &points[j], &points[k], t, K, D)) {
//                     int resIdx = atomicAdd(resultsCount, 1);
//                     results[resIdx * 3] = points[idx].id;
//                     results[resIdx * 3 + 1] = points[j].id;
//                     results[resIdx * 3 + 2] = points[k].id;
//                 }
//             }
//         }
//     }
// }

int computeOnGPU(int *tCount, int *myPointsCount) {
    // Error code to check return values for CUDA calls
    // cudaError_t err = cudaSuccess;

    int numBlocks = (*myPointsCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 gridDim(numBlocks, (*tCount) + 1);
    dim3 blockDim(BLOCK_SIZE);

    return 0;
}


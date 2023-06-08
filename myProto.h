#pragma once

#define BLOCK_SIZE 256

# ifndef POINT_H
# define POINT_H

typedef struct Point {
    int id;
    double x1, x2, a, b;
} Point;

#endif

void test(int *data, int n);
// void checkProximityCriteria(const struct Point* points, double* tValues, int N, int K, double D, int* results, int* resultsCount);
// int satisfyProximityCriteria(const struct Point* p1, const struct Point* p2, const struct Point* p3, double t, int K, double D);
int computeOnGPU(int *tCount, int *myPointsCount);

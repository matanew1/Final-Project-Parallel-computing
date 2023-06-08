#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include "myProto.h"

/*
Simple MPI+OpenMP+CUDA Integration example
Initially the array of size 4*PART is known for the process 0.
It sends the half of the array to the process 1.
Both processes start to increment members of thier members by 1 - partially with OpenMP, partially with CUDA
The results is send from the process 1 to the process 0, which perform the test to verify that the integration worked properly
*/

int main(int argc, char *argv[])
{
   int rank, size;
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   if (argc < 1)
   {
      if (rank == 0)
      {
         fprintf(stderr, "Usage: %s input.txt\n", argv[0]);
      }
      MPI_Finalize();
      return 1;
   }

   int N, K, tCount;
   double D;
   Point *points = NULL;
   points = (Point *)malloc(N * sizeof(Point));
   if (!points)
   {
      printf("malloc failed");
   }

   if (rank == 0)
   {
      FILE *file = fopen("input.txt", "r");
      if (!file)
      {
         fprintf(stderr, "Failed to open input file.\n");
         MPI_Finalize();
         return 1;
      }

      fscanf(file, "%d %d %lf %d\n", &N, &K, &D, &tCount);

      for (int i = 0; i < N; ++i)
      {
         fscanf(file, "%d %lf %lf %lf %lf\n", &points[i].id, &points[i].x1, &points[i].x2, &points[i].a, &points[i].b);
      }

      fclose(file);
   }

   MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&D, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Bcast(&tCount, 1, MPI_INT, 0, MPI_COMM_WORLD);

   // define MPI_POINT constants
   MPI_Datatype MPI_POINT;
   MPI_Type_contiguous(sizeof(Point), MPI_BYTE, &MPI_POINT);
   MPI_Type_commit(&MPI_POINT);

   MPI_Bcast(points, N, MPI_POINT, 0, MPI_COMM_WORLD);

   int pointsPerProcess = N / size;
   int remainingPoints = N % size;
   int myPointsCount = (rank < remainingPoints) ? pointsPerProcess + 1 : pointsPerProcess;
   int myPointsOffset = rank * pointsPerProcess + ((rank < remainingPoints) ? rank : remainingPoints);

   Point *myPoints = (Point *)malloc(myPointsCount * sizeof(Point));
   MPI_Scatter(points, myPointsCount, MPI_POINT,
               myPoints, myPointsCount, MPI_POINT,
               0, MPI_COMM_WORLD);

   double *tValues = (double *)malloc((tCount + 1) * sizeof(double));

// calculate all t points of each  p point
#pragma omp parallel for
   for (int i = 0; i <= tCount; ++i)
   {
      tValues[i] = 2.0 * i / tCount - 1.0;
   }

   int maxResults = 3; // Maximum number of results to find
   int *results = (int *)malloc(maxResults * 3 * sizeof(int));
   int resultsCount = 0;

   computeOnGPU(&tCount, &myPointsCount);

   // int blockSize = 256;
   // int numBlocks = (myPointsCount + blockSize - 1) / blockSize;
   // dim3 gridDim(numBlocks, tCount + 1);
   // dim3 blockDim(blockSize);

   // Point *dPoints;
   // double *dTValues;
   // int *dResults;
   // int *dResultsCount;

   // cudaMalloc(&dPoints, myPointsCount * sizeof(Point));
   // cudaMalloc(&dTValues, (tCount + 1) * sizeof(double));
   // cudaMalloc(&dResults, maxResults * 3 * sizeof(int));
   // cudaMalloc(&dResultsCount, sizeof(int));

   // cudaMemcpy(dPoints, myPoints, myPointsCount * sizeof(Point), cudaMemcpyHostToDevice);
   // cudaMemcpy(dTValues, tValues, (tCount + 1) * sizeof(double), cudaMemcpyHostToDevice);
   // cudaMemset(dResultsCount, 0, sizeof(int));

   // checkProximityCriteria<<<gridDim, blockDim>>>(dPoints, dTValues, myPointsCount, K, D, dResults, dResultsCount);

   // cudaMemcpy(&resultsCount, dResultsCount, sizeof(int), cudaMemcpyDeviceToHost);
   // cudaMemcpy(results, dResults, resultsCount * 3 * sizeof(int), cudaMemcpyDeviceToHost);

   // cudaFree(dPoints);
   // cudaFree(dTValues);
   // cudaFree(dResults);
   // cudaFree(dResultsCount);

   // int *recvCounts = NULL;
   // int *displacements = NULL;
   // int *allResults = NULL;
   // int *allResultsCounts = NULL;
   // int *allResultsDispls = NULL;
   // int *allResultsRecvCounts = NULL;

   // if (rank == 0)
   // {
   //    recvCounts = (int *)malloc(size * sizeof(int));
   //    displacements = (int *)malloc(size * sizeof(int));

   //    allResultsCounts = (int *)malloc(size * sizeof(int));
   //    allResultsDispls = (int *)malloc(size * sizeof(int));
   //    allResultsRecvCounts = (int *)malloc(size * sizeof(int));

   //    for (int i = 0; i < size; ++i)
   //    {
   //       int pointsCount = (i < remainingPoints) ? pointsPerProcess + 1 : pointsPerProcess;
   //       recvCounts[i] = pointsCount * maxResults;
   //       displacements[i] = i * pointsPerProcess + ((i < remainingPoints) ? i : remainingPoints);

   //       allResultsCounts[i] = recvCounts[i];
   //       allResultsDispls[i] = displacements[i] * 3;
   //       allResultsRecvCounts[i] = allResultsCounts[i] * 3;
   //    }

   //    int totalCount = 0;
   //    for (int i = 0; i < size; ++i)
   //    {
   //       totalCount += recvCounts[i];
   //    }

   //    allResults = (int *)malloc(totalCount * sizeof(int));
   // }

   // MPI_Gatherv(results, resultsCount * 3, MPI_INT,
   //             allResults, allResultsRecvCounts, allResultsDispls, MPI_INT,
   //             0, MPI_COMM_WORLD);

   // if (rank == 0)
   // {
   //    FILE *file = fopen("output.txt", "w");
   //    if (!file)
   //    {
   //       fprintf(stderr, "Failed to open output file.\n");
   //       MPI_Finalize();
   //       return 1;
   //    }

   //    if (resultsCount == 0)
   //    {
   //       fprintf(file, "There were no 3 points found for any t.\n");
   //    }
   //    else
   //    {
   //       for (int i = 0; i < resultsCount; ++i)
   //       {
   //          int p1 = allResults[i * 3];
   //          int p2 = allResults[i * 3 + 1];
   //          int p3 = allResults[i * 3 + 2];
   //          double t = tValues[displacements[i]];

   //          fprintf(file, "Points %d, %d, %d satisfy Proximity Criteria at t = %lf\n", p1, p2, p3, t);
   //       }
   //    }

   //    fclose(file);
   // }

   free(points);
   // free(myPoints);
   free(tValues);
   free(results);
   // free(recvCounts);
   // free(displacements);
   // free(allResults);
   // free(allResultsCounts);
   // free(allResultsDispls);
   // free(allResultsRecvCounts);

   MPI_Finalize();
   return 0;
}

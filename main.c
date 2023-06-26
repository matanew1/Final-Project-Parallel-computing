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
   double *tValues = (double *)malloc((tCount) * sizeof(double));

   if (rank == 0)
   {
      FILE *file = fopen("input.txt", "r");
      if (!file)
      {
         fprintf(stderr, "Failed to open input file.\n");
         MPI_Finalize();
         return 1;
      }

      if (fscanf(file, "%d %d %lf %d\n", &N, &K, &D, &tCount) != 4)
      {
         fprintf(stderr, "Failed to read required values from input file.\n");
         fclose(file);
         MPI_Finalize();
         return 1;
      }
      points = (Point *)malloc(N * sizeof(Point));
      if (!points)
      {
         fprintf(stderr, "Failed to allocate points.\n");
         MPI_Finalize();
         return 1;
      }

      for (int i = 0; i < N; ++i)
      {
         if (fscanf(file, "%d %lf %lf %lf %lf\n", &points[i].id, &points[i].x1, &points[i].x2, &points[i].a, &points[i].b) != 5)
         {
            fprintf(stderr, "Failed to read point data from input file.\n");
            fclose(file);
            MPI_Finalize();
            return 1;
         }
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

   // calculate all t points
#pragma omp parallel for
   for (int i = 0; i <= tCount; ++i)
   {
      tValues[i] = 2.0 * i / tCount - 1.0;
   }
   int maxResults = 3; // Maximum number of results to find
   int *results = (int *)malloc(maxResults * 3 * sizeof(int));
   int resultsCount = 0;

   int tPointsPerProcess = tCount / size;
   int remainingTPoints = tCount % size;
   int myTPointsCount = (rank < remainingTPoints) ? tPointsPerProcess + 1 : tPointsPerProcess;
   int myTPointsOffset = rank * tPointsPerProcess + ((rank < remainingTPoints) ? rank : remainingTPoints);

   double *myTPoints = (double *)malloc(myTPointsCount * sizeof(double));
   MPI_Scatter(tValues, myTPointsCount, MPI_INT, myTPoints, myTPointsCount, MPI_INT, 0, MPI_COMM_WORLD);

   printf("rank = %d dount=%d offset=%d\n",rank,myTPointsCount,myTPointsOffset);
   for(int i = 0; i <)
   



   /*
      N = 4
      K = 2
      D = 1.23
      tCount = 100
      myTPointsCount = 2, 2 points for 1 process
      points = array of Point struct
      results = empty array of results 
      resultsCount = 0
      maxResults = 3
      myTPoints = array point of each process
   */
   computeOnGPU(&N, &K, &D, &tCount, &myTPointsCount, tValues, results, &resultsCount, &maxResults, myTPoints);

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
   //       int pointsCount = (i < remainingTPoints) ? tPointsPerProcess + 1 : tPointsPerProcess;
   //       recvCounts[i] = pointsCount * maxResults;
   //       displacements[i] = i * tPointsPerProcess + ((i < remainingTPoints) ? i : remainingTPoints);

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
   free(myTPoints);
   free(tValues);
   free(results);
   MPI_Type_free(&MPI_POINT);
   // free(recvCounts);
   // free(displacements);
   // free(allResults);
   // free(allResultsCounts);
   // free(allResultsDispls);
   // free(allResultsRecvCounts);

   MPI_Finalize();
   return 0;
}

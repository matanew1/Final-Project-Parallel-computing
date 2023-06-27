#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include "myProto.h"

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
         fclose(file); // Close the file before returning
         MPI_Finalize();
         return 1;
      }

      for (int i = 0; i < N; ++i)
      {
         if (fscanf(file, "%d %lf %lf %lf %lf\n", &points[i].id, &points[i].x1, &points[i].x2, &points[i].a, &points[i].b) != 5)
         {
            fprintf(stderr, "Failed to read point data from input file.\n");
            fclose(file); // Close the file before returning
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

   if (rank != 0)
   {
      points = (Point *)malloc(N * sizeof(Point));
      if (!points)
      {
         fprintf(stderr, "Failed to allocate points.\n");
         MPI_Finalize();
         return 1;
      }
   }

   MPI_Bcast(points, N, MPI_POINT, 0, MPI_COMM_WORLD);

   double *tValues = (double *)malloc(tCount * sizeof(double));
   // calculate all t points
#pragma omp parallel for
   for (int i = 0; i < tCount; ++i)
   {
      tValues[i] = 2.0 * i / tCount - 1.0;
   }

   int mytValuesSize = tCount / size;
   int remainingTValues = tCount % size;
   int *sendcounts = (int *)malloc(size * sizeof(int));
   int *senddispls = (int *)malloc(size * sizeof(int));

   // Calculate the sendcounts and displacements
   for (int i = 0; i < size; i++)
   {
      sendcounts[i] = (i < remainingTValues) ? mytValuesSize + 1 : mytValuesSize;
      senddispls[i] = i * mytValuesSize + ((i < remainingTValues) ? i : remainingTValues);
   }
   int tCountSize = sendcounts[rank];
   double *myTValues = (double *)malloc(tCountSize * sizeof(double));

   MPI_Scatterv(tValues, sendcounts, senddispls, MPI_DOUBLE, myTValues, tCountSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   int count = 0;
   int globalCount = 0;

   int **results = (int **)malloc(N * sizeof(int *));
   for (int i = 0; i < N; i++)
   {
      results[i] = (int *)malloc(tCountSize * sizeof(int));
      for (int j = 0; j < tCountSize; j++)
      {
         results[i][j] = -1;
      }
   }


   computeOnGPU(&count, &N, &K, &D, &tCountSize, myTValues, points, results);

   // Gather results from all processes into global_results on rank 0
   int* recvcounts = (int*)malloc(size * sizeof(int));
   int* recvdispls = (int*)malloc(size * sizeof(int));

   // Calculate the recvcounts and displacements for the 2D array
   for (int i = 0; i < size; i++)
   {
      recvcounts[i] = N * tCountSize;
      recvdispls[i] = 0;
   }
   

   // Adjust the displacements for the 2D array
   for (int i = 1; i < size; i++)
   {
      recvdispls[i] = recvdispls[i - 1] + recvcounts[i - 1];
   }

   int** global_results = NULL;
   if (rank == 0)
   {
      global_results = (int**)malloc(N * sizeof(int*));
      for (int i = 0; i < N; i++)
      {
         global_results[i] = (int*)malloc(tCount * sizeof(int));
         for (int j = 0; j < tCount; j++)
         {
            global_results[i][j] = -1;
         }
         
      }
   }

   // Gather the 2D array results from all processes into global_results on rank 0
   MPI_Gatherv(&(results[0][0]), N * tCountSize, MPI_INT,
               &(global_results[0][0]), recvcounts, recvdispls, MPI_INT,
               0, MPI_COMM_WORLD);

   if (rank == 0)
   {
      printf("Global Count: %d\n", globalCount);

      // Deallocate global_results memory
      for (int i = 0; i < N; i++)
      {
         free(global_results[i]);
      }
      free(global_results);
   }

   // Deallocate memory
   free(points);
   free(tValues);
   MPI_Type_free(&MPI_POINT);
   free(sendcounts);
   free(recvdispls);
   free(myTValues);
   for (int i = 0; i < N; i++)
   {
      free(results[i]);
   }
   free(results);

   MPI_Finalize();
   return 0;
}

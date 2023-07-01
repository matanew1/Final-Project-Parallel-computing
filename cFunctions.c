#include "myProto.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void readInputFile(const char *filename, int *N, int *K, double *D, int *tCount, Point **points)
{
   FILE *file = fopen(filename, "r");
   if (!file)
   {
      fprintf(stderr, "Failed to open input file.\n");
      MPI_Finalize();
      exit(1);
   }

   if (fscanf(file, "%d %d %lf %d\n", N, K, D, tCount) != 4)
   {
      fprintf(stderr, "Failed to read required values from input file.\n");
      fclose(file);
      MPI_Finalize();
      exit(1);
   }

   *points = (Point *)malloc(*N * sizeof(Point));
   if (!(*points))
   {
      fprintf(stderr, "Failed to allocate points.\n");
      fclose(file);
      MPI_Finalize();
      exit(1);
   }

   for (int i = 0; i < *N; ++i)
   {
      if (fscanf(file, "%d %lf %lf %lf %lf\n", &((*points)[i].id), &((*points)[i].x1), &((*points)[i].x2), &((*points)[i].a), &((*points)[i].b)) != 5)
      {
         fprintf(stderr, "Failed to read point data from input file.\n");
         fclose(file);
         MPI_Finalize();
         exit(1);
      }
   }

   fclose(file);
}

void calculateTValues(int tCount, double **tValues)
{
   *tValues = (double *)malloc((tCount + 1) * sizeof(double));

   // calculate all t points
#pragma omp parallel for
   for (int i = 0; i <= tCount; ++i)
   {
      (*tValues)[i] = (2.0 * i / tCount) - 1;
   }
}

void gatherResults(int rank, int size, int N, int tCount, int tCountSize, int *results, int *global_results)
{
   int *recvcounts = (int *)malloc(size * sizeof(int));
   int *displs = (int *)malloc(size * sizeof(int));

   // Calculate the recvcounts for the 2D array
   for (int i = 0; i < size; i++)
   {
      recvcounts[i] = CONSTRAINTS * ((i < tCount % size) ? (tCount / size + 1) : (tCount / size));
   }

   // Calculate the displacements for the 2D array
   displs[0] = 0;
   for (int i = 1; i < size; i++)
   {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
   }

   // Gather the 2D array results from all processes into global_results on rank 0
   MPI_Gatherv(results, CONSTRAINTS * tCountSize, MPI_INT,
               global_results, recvcounts, displs, MPI_INT,
               0, MPI_COMM_WORLD);

   free(recvcounts);
   free(displs);
}

void writeOutputFile(const char *filename, int tCount, int *results, Point *points, int N)
{
   FILE *file = fopen(filename, "w");
   if (!file)
   {
      fprintf(stderr, "Failed to open output file.\n");
      MPI_Finalize();
      exit(1);
   }

   int proximityFound = 0;

   for (int i = 0; i < tCount; i++)
   {
      int count = 0;
      int pointIDs[3] = {-1, -1, -1};

      for (int j = 0; j < CONSTRAINTS; j++)
      {
         int pointID = results[i * CONSTRAINTS + j];
         if (pointID >= 0 && pointID < N)
         {
            pointIDs[count] = pointID;
            count++;
         }
      }

      if (count == 3)
      {
         proximityFound = 1;
         fprintf(file, "Points ");
         for (int j = 0; j < 3; j++)
         {
            fprintf(file, "pointID%d", pointIDs[j]);
            if (j < 2)
               fprintf(file, ", ");
         }
         fprintf(file, " satisfy Proximity Criteria at t = %.2f\n", 2.0 * i / tCount - 1);
      }
   }

   if (!proximityFound)
   {
      fprintf(file, "There were no 3 points found for any t.\n");
   }

   fclose(file);
}

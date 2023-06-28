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

   const char *filename = "input.txt"; // Predefined filename

   int N, K, tCount;
   double D;
   Point *points = NULL;

   if (rank == 0)
   {
      readInputFile(filename, &N, &K, &D, &tCount, &points);
   }

   MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&D, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Bcast(&tCount, 1, MPI_INT, 0, MPI_COMM_WORLD);

   // Define MPI_POINT constants
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
         exit(1);
      }
   }

   MPI_Bcast(points, N, MPI_POINT, 0, MPI_COMM_WORLD);

   double *tValues = NULL;
   calculateTValues(tCount, &tValues);

   int tCountSize = tCount / size;
   int remainingTValues = tCount % size;
   int *sendcounts = (int *)malloc(size * sizeof(int));
   int *displs = (int *)malloc(size * sizeof(int));

   // Calculate the sendcounts and displacements
   for (int i = 0; i < size; i++)
   {
      sendcounts[i] = (i < remainingTValues) ? tCountSize + 1 : tCountSize;
      displs[i] = i * tCountSize + ((i < remainingTValues) ? i : remainingTValues);
   }
   int myTValuesSize = sendcounts[rank];
   double *myTValues = (double *)malloc(myTValuesSize * sizeof(double));

   MPI_Scatterv(tValues, sendcounts, displs, MPI_DOUBLE, myTValues, myTValuesSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   int count = 0;
   int globalCount = 0;

   int *results = (int *)malloc(CONSTRAINTS * tCountSize * sizeof(int));
   for (int i = 0; i < CONSTRAINTS * tCountSize; i++)
      results[i] = -1;

   // Compute results on GPU
   computeOnGPU(&N, &K, &D, &myTValuesSize, myTValues, points, results);

   int *global_results = NULL;
   if (rank == 0)
   {
      global_results = (int *)malloc(CONSTRAINTS * tCount * sizeof(int));
      for (int i = 0; i < CONSTRAINTS * tCount; i++)
      {
         global_results[i] = -1;         
      }
   }
   
   gatherResults(rank, size, N, tCount, myTValuesSize, results, global_results);

   if (rank == 0)
   {
      writeOutputFile("output.txt", tCount, global_results, points);

      // Deallocate global_results memory
      free(global_results);
   }

   // Deallocate memory
   free(points);
   free(tValues);
   MPI_Type_free(&MPI_POINT);
   free(sendcounts);
   free(displs);
   free(myTValues);
   free(results);

   MPI_Finalize();
   return 0;
}

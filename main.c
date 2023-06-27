#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include "myProto.h"

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
   *tValues = (double *)malloc(tCount * sizeof(double));

   // calculate all t points
#pragma omp parallel for
   for (int i = 0; i < tCount; ++i)
   {
      (*tValues)[i] = 2.0 * i / tCount - 1.0;
   }
}

void gatherResults(int rank, int size, int N, int tCount, int tCountSize, int **results, int ***global_results)
{
   int *recvcounts = (int *)malloc(size * sizeof(int));
   int *displs = (int *)malloc(size * sizeof(int));

   // Calculate the recvcounts and displacements for the 2D array
   for (int i = 0; i < size; i++)
   {
      recvcounts[i] = N * tCountSize;
      displs[i] = 0;
   }

   // Adjust the displacements for the 2D array
   for (int i = 1; i < size; i++)
   {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
   }

   if (rank == 0)
   {
      *global_results = (int **)malloc(N * sizeof(int *));
      for (int i = 0; i < N; i++)
      {
         (*global_results)[i] = (int *)malloc(tCount * sizeof(int));
         for (int j = 0; j < tCount; j++)
         {
            (*global_results)[i][j] = -1;
         }
      }
   }

   printf("rank = %d recvcounts = %d",rank, recvcounts[rank]);
   printf("rank = %d displs = %d",rank, displs[rank]);
   for (int i = 0; i < N; i++)
   {
      for (int j = 0; j < tCountSize; j++)
      {
         printf("%d ",results[i][j]);
      }
      printf("\n");
   }
   printf("\n");

   // Gather the 2D array results from all processes into global_results on rank 0
   // MPI_Gatherv(&(results[0][0]), N * tCountSize, MPI_INT,
   //             &((*global_results)[0][0]), recvcounts, displs, MPI_INT,
   //             0, MPI_COMM_WORLD);

   free(recvcounts);
   free(displs);
}

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

   int **results = (int **)malloc(N * sizeof(int *));
   for (int i = 0; i < N; i++)
   {
      results[i] = (int *)malloc(myTValuesSize * sizeof(int));
      for (int j = 0; j < myTValuesSize; j++)
      {
         results[i][j] = -1;
      }
   }

   // Compute results on GPU
   computeOnGPU(&count, &N, &K, &D, &myTValuesSize, myTValues, points, results);

   // Reduce the local count to get the global count
   MPI_Reduce(&count, &globalCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

   int **global_results = NULL;
   gatherResults(rank, size, N, tCount, myTValuesSize, results, &global_results);

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
   free(displs);
   free(myTValues);
   for (int i = 0; i < N; i++)
   {
      free(results[i]);
   }
   free(results);

   MPI_Finalize();
   return 0;
}

/*
mpiCudaOpemMP:39929 terminated with signal 11 at PC=56079a8f0b91 SP=7ffeb0cee360.  Backtrace:
./mpiCudaOpemMP(+0x2b91)[0x56079a8f0b91]
./mpiCudaOpemMP(+0x309e)[0x56079a8f109e]
/lib/x86_64-linux-gnu/libc.so.6(+0x29d90)[0x7f656f429d90]
/lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x80)[0x7f656f429e40]
./mpiCudaOpemMP(+0x2585)[0x56079a8f0585]
make: *** [Makefile:12: run] Error 1
*/

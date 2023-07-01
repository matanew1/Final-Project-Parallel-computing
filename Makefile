NUM_PROC = 4
build:
	gcc ./test.c -o test -lm
	mpicxx -fopenmp -c main.c -o main.o
	mpicxx -fopenmp -c cFunctions.c -o cFunctions.o
	nvcc -I./Common  -gencode arch=compute_61,code=sm_61 -c cudaFunctions.cu -o cudaFunctions.o
	mpicxx -fopenmp -o mpiCudaOpemMP  main.o cFunctions.o cudaFunctions.o  -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart
	

clean:
	rm -f *.o ./mpiCudaOpemMP

run:
	./test
	mpiexec -np ${NUM_PROC} ./mpiCudaOpemMP

runOn2:
	mpiexec -np ${NUM_PROC} -machinefile  mf  -map-by  node  ./mpiCudaOpemMP

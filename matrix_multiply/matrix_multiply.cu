//
// matrix_multiply.cu
//
// matrix multiplication with CUDA multithreading
//

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// DEFAULT_DIM
//
// the default matrix dimension
//
#define DEFAULT_DIM 16

// CUDA_ERROR
//
// CUDA error check
//
#define CUDA_ERROR(error) { cudaGpuErrorAssert((error), __FILE__, __LINE__); }
inline
void
cudaGpuErrorAssert (
  cudaError_t  error,
  const char  *file,
  int          line
) {
  if (error != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(error), file, line);
    exit(error);
  }
}

// cuda_multiply_matrix
//
// CUDA kernel for multiplying matrix
//
__global__
void
cuda_multiply_matrix (
  int *result,
  int *a,
  int *b,
  int  dim
) {
  // multiply each cell of this row with each cell of the column
  int value = 0;
  for (int k = 0; k < dim; ++k) {
    value += a[blockIdx.x * dim + k] * b[k * dim + blockIdx.y];
  }
  result[blockIdx.x * dim + blockIdx.y] = value;
}

// gpu_multiply_matrix
//
// multiply matrices using the GPU
//
__host__
void
gpu_multiply_matrix (
  int *result,
  int *a,
  int *b,
  int  dim
) {
  int *result_d;
  int *a_d;
  int *b_d;
  // get the size of the matrix to allocate
  int size = dim * dim * sizeof(int);
  // allocate the memory on the GPU
  CUDA_ERROR(cudaMalloc(&result_d, size));
  CUDA_ERROR(cudaMalloc(&a_d, size));
  CUDA_ERROR(cudaMalloc(&b_d, size));
  // copy the memory to the GPU
  CUDA_ERROR(cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice));
  CUDA_ERROR(cudaMemcpy(b_d, b, size, cudaMemcpyHostToDevice));
  // start GPU kernel with one thread for each cell of the result matrix
  cuda_multiply_matrix<<<dim3(dim, dim), 1>>>(result_d, a_d, b_d, dim);
  // copy memory back from the GPU
  CUDA_ERROR(cudaMemcpy(result, result_d, size, cudaMemcpyDeviceToHost));
}

// cpu_multiply_matrix
//
// multiply matrices using the CPU
//
__host__
void
cpu_multiply_matrix (
  int *result,
  int *a,
  int *b,
  int  dim
) {
  // multiply matrices
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      int value = 0;
      for (int k = 0; k < dim; ++k) {
        value += a[i * dim + k] * b[k * dim + j];
      }
      result[i * dim + j] = value;
    }
  }
}

// allocate_matrix
//
// allocate a matrix of square dimensions
//
__host__
int *
allocate_matrix (
  int dim
) {
  int *m;
  // check matrix size
  if (dim <= 0) {
    return NULL;
  }
  // allocate matrix
  m = (int *)malloc(dim * dim * sizeof(int));
  if (m == NULL) {
    perror("Failed to allocate memory for matrix");
    exit(errno);
  }
  return m;
}

// generate_matrix
//
// generate a matrix of square dimensions with random 0 or 1 values in cells
//
__host__
int *
generate_matrix (
  int dim
) {
  // allocate a matrix
  int *m = allocate_matrix(dim);
  if (m != NULL) {
    // generate random values 0 or 1 for each cell
    for (int i = 0; i < dim; ++i) {
      for (int j = 0; j < dim; ++j) {
        m[i * dim + j] = rand() % 2;
      }
    }
  }
  return m;
}

// print_matrix
//
// print matrix of square dimension
//
__host__
void
print_matrix (
  int *m,
  int  dim
) {
  for (int i = 0; i < dim; ++i) {
    printf("[");
    for (int j = 0; j < dim; ++j) {
      printf(" % 2.2d", m[i * dim + j]);
    }
    printf("]\n");
  }
}

// main
//
// multiply matrices using the CPU and GPU with time measurement
//
int
main (
  int   argc,
  char *argv[]
) {
  cudaEvent_t  cpu_start;
  cudaEvent_t  cpu_finish;
  cudaEvent_t  gpu_start;
  cudaEvent_t  gpu_finish;
  float        measurement;
  int          verbose = 0;
  int          dim = DEFAULT_DIM;
  int         *result;
  int         *a;
  int         *b;
  // parse arguments
  for (int i = 1; i < argc; ++i) {
    if ((strcasecmp(argv[i], "-v") == 0) || (strcasecmp(argv[i], "--verbose") == 0)) {
      // verbose enabled
      verbose = 1;
    } else {
      // dimension specified
      dim = atoi(argv[i]);
      if (dim <= 0) {
        printf("Usage: matrix_multiply [-v/--verbose] [dimension=%d]\nMultiply a square matrix of specified dimension\n", DEFAULT_DIM);
        return 0;
      }
    }
  }
  // create start and finish events
  CUDA_ERROR(cudaEventCreate(&cpu_start));
  CUDA_ERROR(cudaEventCreate(&cpu_finish));
  CUDA_ERROR(cudaEventCreate(&gpu_start));
  CUDA_ERROR(cudaEventCreate(&gpu_finish));
  // generate matrices randomly
  result = allocate_matrix(dim);
  a = generate_matrix(dim);
  b = generate_matrix(dim);
  // print matrices
  if (verbose != 0) {
    printf("A:\n");
    print_matrix(a, dim);
    printf("B:\n");
    print_matrix(b, dim);
  }
  // get start time
  CUDA_ERROR(cudaEventRecord(cpu_start));
  // multiply the matrices on the CPU
  cpu_multiply_matrix(result, a, b, dim);
  // get finish time
  CUDA_ERROR(cudaEventRecord(cpu_finish));
  CUDA_ERROR(cudaEventSynchronize(cpu_finish));
  // print result
  if (verbose != 0) {
    printf("CPU Result:\n");
    print_matrix(result, dim);
  }
  // print time measurement
  CUDA_ERROR(cudaEventElapsedTime(&measurement, cpu_start, cpu_finish));
  printf("CPU Time: %f\n", measurement / 1000.0);
  // get start time
  CUDA_ERROR(cudaEventRecord(gpu_start));
  // multiple the matrices on the GPU
  gpu_multiply_matrix(result, a, b, dim);
  // get finish time
  CUDA_ERROR(cudaEventRecord(gpu_finish));
  CUDA_ERROR(cudaEventSynchronize(gpu_finish));
  // print result
  if (verbose != 0) {
    printf("GPU Result:\n");
    print_matrix(result, dim);
  }
  // print time measurement
  CUDA_ERROR(cudaEventElapsedTime(&measurement, gpu_start, gpu_finish));
  printf("GPU Time: %f\n", measurement / 1000.0);
  return 0;
}

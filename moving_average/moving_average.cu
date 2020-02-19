//
// moving_average.cu
//
// calculate moving averages with CUDA multithreading
//

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>


// DEFAULT_RAND
//
// the default random max
//
#define DEFAULT_RAND 100.00f

// DEFAULT_SIZE
//
// the default array size
//
#define DEFAULT_SIZE 10000

// DEFAULT_N
//
// the default sample size
//
#define DEFAULT_N 16

// DEFAULT_DIM_GRID_X
//
// the default grid dimensions
//
#define DEFAULT_DIM_GRID_X 10
// DEFAULT_DIM_GRID_Y
//
// the default grid dimensions
//
#define DEFAULT_DIM_GRID_Y 1
// DEFAULT_DIM_GRID_Z
//
// the default grid dimensions
//
#define DEFAULT_DIM_GRID_Z 1

// DEFAULT_DIM_BLOCKS_X
//
// the default block dimensions
//
#define DEFAULT_DIM_BLOCKS_X 32
// DEFAULT_DIM_BLOCKS_Y
//
// the default block dimensions
//
#define DEFAULT_DIM_BLOCKS_Y 32
// DEFAULT_DIM_BLOCKS_Z
//
// the default block dimensions
//
#define DEFAULT_DIM_BLOCKS_Z 1

// DEFAULT_WIDTH
//
// the default width of numbers to print
//
#define DEFAULT_WIDTH 16

// CUDA_ERROR
//
// CUDA error check
//
#define CUDA_ERROR(error) { cudaGpuErrorAssert((error), __FILE__, __LINE__); }
inline void cudaGpuErrorAssert(cudaError_t error, const char *file, int line)
{
  if (error != cudaSuccess) {
     fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(error), file, line);
     exit(error);
  }
}

// cuda_moving_average
//
// CUDA kernel for calculating moving average
//
__global__
void
cuda_moving_average (
  float *result,
  float *numbers,
  int    size,
  int    n
) {
  // shared result
  extern __shared__ float result_s[];
  // get offset and check it is valid
  int offset = blockIdx.x * DEFAULT_DIM_GRID_Y + blockIdx.y;
  offset *= DEFAULT_DIM_GRID_Z;
  offset += blockIdx.z;
  offset *= DEFAULT_DIM_BLOCKS_X;
  offset += threadIdx.x;
  offset *= DEFAULT_DIM_BLOCKS_Y;
  offset += threadIdx.y;
  offset *= DEFAULT_DIM_BLOCKS_Z;
  offset += threadIdx.z;
  int threadOffset = threadIdx.x * DEFAULT_DIM_BLOCKS_Y + threadIdx.y;
  threadOffset *= DEFAULT_DIM_BLOCKS_Z;
  threadOffset += threadIdx.z;
  if ((offset < size) && (threadOffset < (DEFAULT_DIM_BLOCKS_X * DEFAULT_DIM_BLOCKS_Y * DEFAULT_DIM_BLOCKS_Z))) {
    // load each element and sync threads
    result_s[threadOffset] = numbers[offset];
    // copy the additional elements needed for the last average
    if (threadOffset == 0) {
      int to = DEFAULT_DIM_BLOCKS_X * DEFAULT_DIM_BLOCKS_Y * DEFAULT_DIM_BLOCKS_Z;
      for (int i = 0; i < n; ++i) {
        int o = offset + to + i;
        if (o < size) {
          result_s[to + i] = numbers[o];
        }
      }
    }
    __syncthreads();
    // calculate the moving average
    float value = 0.0;
    for (int k = 0; k < n; ++k) {
      value += result_s[threadOffset + k];
    }
    value /= n;
    // store the element and sync threads
    result[offset] = value;
  }
}

// gpu_moving_average
//
// calculate moving averages using the GPU
//
__host__
void
gpu_moving_average (
  float *result,
  float *numbers,
  int    size,
  int    n
) {
  float *result_d;
  float *numbers_d;
  // get the size of the array to allocate
  int bytesize = size * sizeof(float);
  // allocate the memory on the GPU
  CUDA_ERROR(cudaMalloc(&result_d, bytesize));
  CUDA_ERROR(cudaMalloc(&numbers_d, bytesize));
  // copy the memory to the GPU
  CUDA_ERROR(cudaMemcpy(numbers_d, numbers, size, cudaMemcpyHostToDevice));
  // start GPU kernel with one thread for each element of the result array
  dim3 dimGrid(DEFAULT_DIM_GRID_X, DEFAULT_DIM_GRID_Y, DEFAULT_DIM_GRID_Z);
  dim3 dimBlocks(DEFAULT_DIM_BLOCKS_X, DEFAULT_DIM_BLOCKS_Y, DEFAULT_DIM_BLOCKS_Z);
  int sharedSize = sizeof(float) * ((DEFAULT_DIM_BLOCKS_X * DEFAULT_DIM_BLOCKS_Y * DEFAULT_DIM_BLOCKS_Z) + n);
  cuda_moving_average<<<dimGrid, dimBlocks, sharedSize>>>(result_d, numbers_d, size, n);
  // copy memory back from the GPU
  CUDA_ERROR(cudaMemcpy(result, result_d, bytesize, cudaMemcpyDeviceToHost));
}

// allocate_array
//
// allocate an array
//
__host__
float *
allocate_array (
  int size
) {
  float *a;
  // check array size
  if (size <= 0) {
    return NULL;
  }
  // allocate array
  a = (float *)malloc(size * sizeof(float));
  if (a == NULL) {
    perror("Failed to allocate memory for array");
    exit(errno);
  }
  return a;
}

// generate_array
//
// generate an array of random numbers
//
__host__
float *
generate_array (
  int size
) {
  // allocate a matrix
  float *numbers = allocate_array(size);
  if (numbers != NULL) {
    // generate random values for each element
    for (int i = 0; i < size; ++i) {
      numbers[i] = ((float)rand()/(float)(RAND_MAX)) * DEFAULT_RAND;
    }
  }
  return numbers;
}

// print_array
//
// print array
//
__host__
void
print_array (
  float *numbers,
  int    size
) {
  for (int i = 0; i < size; ++i) {
    if ((i % DEFAULT_WIDTH) == 0) {
      printf("%6.6d: ", i);
    }
    printf(" %4.2f", numbers[i]);
    if ((i % DEFAULT_WIDTH) == (DEFAULT_WIDTH - 1)) {
      printf("\n");
    }
  }
  if ((size % DEFAULT_WIDTH) != (DEFAULT_WIDTH - 1)) {
    printf("\n");
  }
}

// main
//
// calculate moving averages using the GPU with time measurement
//
int
main (
  int   argc,
  char *argv[]
) {
  cudaEvent_t  start;
  cudaEvent_t  finish;
  float        measurement;
  int          size = DEFAULT_SIZE;
  int          n = DEFAULT_N;
  float       *result;
  float       *numbers;
  // seed the random number generator
  srand(time(NULL));
  // create start and finish events
  CUDA_ERROR(cudaEventCreate(&start));
  CUDA_ERROR(cudaEventCreate(&finish));
  // print settings being used
  printf("Size: %d\nN: %d\n", DEFAULT_SIZE, DEFAULT_N);
  printf("Kernel: <<<dim3(%d, %d, %d), dim3(%d, %d, %d)>>>\n",
         DEFAULT_DIM_GRID_X, DEFAULT_DIM_GRID_Y, DEFAULT_DIM_GRID_Z,
         DEFAULT_DIM_BLOCKS_X, DEFAULT_DIM_BLOCKS_Y, DEFAULT_DIM_BLOCKS_Z);
  // generate numbers randomly
  result = allocate_array(size);
  numbers = generate_array(size);
  // print numbers
  printf("Numbers[%d]:\n", size);
  print_array(numbers, size);
  // get start time
  CUDA_ERROR(cudaEventRecord(start));
  // calculate the moving averages on the GPU
  gpu_moving_average(result, numbers, size, n);
  // get finish time
  CUDA_ERROR(cudaEventRecord(finish));
  CUDA_ERROR(cudaEventSynchronize(finish));
  // print result
  printf("Result[%d]:\n", size);
  print_array(result, size);
  // print time measurement
  CUDA_ERROR(cudaEventElapsedTime(&measurement, start, finish));
  printf("Time: %f\n", measurement / 1000.0);
  return 0;
}

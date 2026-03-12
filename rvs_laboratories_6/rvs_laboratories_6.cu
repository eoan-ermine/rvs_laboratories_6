#include "wb.h"

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Вычисление C = A * B
#define TILE_SIZE 16  // Аналог шаблонного BLOCK_SIZE

__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Индексы блока и потока
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Глобальные координаты элемента в C
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    // Параметры обхода по внутренней размерности (K = numAColumns)
    int aBegin = numAColumns * TILE_SIZE * by;  // Смещение начала строки блока в A
    int bBegin = TILE_SIZE * bx;                 // Смещение начала столбца блока в B
    int aStep  = TILE_SIZE;
    int bStep  = TILE_SIZE * numBColumns;
    int numTiles = (numAColumns + TILE_SIZE - 1) / TILE_SIZE;

    float Csub = 0.0f;

    for (int t = 0; t < numTiles; ++t) {
        // Загрузка подматрицы A с проверкой границ
        int aCol = t * TILE_SIZE + tx;
        if (row < numARows && aCol < numAColumns)
            As[ty][tx] = A[aBegin + t * aStep + numAColumns * ty + tx];
        else
            As[ty][tx] = 0.0f;

        // Загрузка подматрицы B с проверкой границ
        int bRow = t * TILE_SIZE + ty;
        if (bRow < numBRows && col < numBColumns)
            Bs[ty][tx] = B[bBegin + t * bStep + numBColumns * ty + tx];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Умножение текущих тайлов
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            Csub += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    // Запись результата с проверкой границ
    if (row < numCRows && col < numCColumns)
        C[row * numCColumns + col] = Csub;
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // Матрица A
  float *hostB; // Матрица B
  float *hostC; // Выходная матрица C
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // количество строк матрицы A
  int numAColumns; // количество столбцов матрицы A
  int numBRows;    // количество строк матрицы B
  int numBColumns; // количество столбцов матрицы B
  int numCRows;    // количество строк матрицы  C (установите
                              // это значение сами)
  int numCColumns; // количество столбцов матрицы C (установите
                   //это значение сами)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Установите numCRows и numCColumns
  numCRows    = numARows;
  numCColumns = numBColumns;
  //@@ Выделение памяти под матрицу hostC
  hostC = static_cast<float*>(malloc(numCRows * numCColumns * sizeof(float)));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Выделите память GPU
  cudaMalloc(&deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(float));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Скопируйте память с хоста на GPU
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Инициализируйте размерности блоков и сетки
dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
dim3 numBlocks((numCColumns + TILE_SIZE - 1) / TILE_SIZE,
               (numCRows    + TILE_SIZE - 1) / TILE_SIZE);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ запустите ядро GPU
  matrixMultiply<<<numBlocks, threadsPerBlock>>>(
      deviceA, deviceB, deviceC,
      numARows, numAColumns,
      numBRows, numBColumns,
      numCRows, numCColumns
  );

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Скопируйте память обратно с GPU на хост
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Освободите память GPU
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}

#include <stdio.h>
#include <cuda_runtime.h>


// Definicje funkcji kerneli
__global__ void bitonicSortStep(int *dev_values, int j, int k, int n);
__global__ void bitonicCompareAndSwapEven(int *dev_values, int j, int k, int n);
__global__ void bitonicCompareAndSwapOdd(int *dev_values, int j, int k, int n);
__global__ void evenSortKernel(int *data, int n);
__global__ void oddSortKernel(int *data, int n);

// Definicje funkcji pomocniczych
float bitonicSort(int *values, int n, int **sorted_values);
float bitonicSortprim(int *values, int n, int **sorted_values);
float oddEvenSort(int *values, int n, int **sorted_values);
void printArray(const char *message, int *array, int n);

int main() {
    const int n = 2048;
    int values[n], *sorted_values, *sorted_values1, *sorted_values2;
    float time, time1, time2;

    // Generowanie danych wejściowych
    printf("Dane wejściowe:\n");
    for (int i = 0; i < n; i++) {
        values[i] = rand() % (n + i);
        printf("%d ", values[i]);
    }
    printf("\n");

    // Sortowanie bitoniczne w 2 kernelach
    time = bitonicSortprim(values, n, &sorted_values);
    printf("Czas sortowania bitonicznego w 2 kernelach: %f ms\n", time);
    printArray("Posortowane dane (2 kernele):", sorted_values, n);

    // Sortowanie bitoniczne w 1 kernelu
    time1 = bitonicSort(values, n, &sorted_values1);
    printf("Czas sortowania bitonicznego w 1 kernelu: %f ms\n", time1);
    printArray("Posortowane dane (1 kernel):", sorted_values1, n);

    // Sortowanie odd-even
    time2 = oddEvenSort(values, n, &sorted_values2);
    printf("Czas sortowania odd-even: %f ms\n", time2);
    printArray("Posortowane dane (odd-even):", sorted_values2, n);

    // Sprzątanie
    free(sorted_values);
    free(sorted_values1);
    free(sorted_values2);

    return 0;
}


__global__ void bitonicSortStep(int *dev_values, int j, int k, int n) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;

    if ((ixj) > i) {
        if ((i & k) == 0) {
            if (dev_values[i] > dev_values[ixj]) {
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        } else {
            if (dev_values[i] < dev_values[ixj]) {
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
    }
}

float bitonicSort(int *values, int n, int **sorted_values) {
    int *dev_values;
    size_t size = n * sizeof(int);

    cudaMalloc(&dev_values, size);
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int k, j;
    for (k = 2; k <= n; k <<= 1) {
        for (j = k >> 1; j > 0; j = j >> 1) {
            bitonicSortStep<<<n/k, k>>>(dev_values, j, k, n);
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    *sorted_values = (int*)malloc(size);
    cudaMemcpy(*sorted_values, dev_values, size, cudaMemcpyDeviceToHost);

    cudaFree(dev_values);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}


__global__ void bitonicCompareAndSwapEven(int *dev_values, int j, int k, int n) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;

    // Porównanie i zamiana dla parzystych indeksów
    if ((ixj) > i) {
        if ((i & k) == 0 && i < n) {
            if (dev_values[i] > dev_values[ixj]) {
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
    }
}

__global__ void bitonicCompareAndSwapOdd(int *dev_values, int j, int k, int n) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;

    // Porównanie i zamiana dla nieparzystych indeksów
    if ((ixj) > i) {
        if ((i & k) != 0 && i < n) {
            if (dev_values[i] < dev_values[ixj]) {
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
    }
}

float bitonicSortprim(int *values, int n, int **sorted_values) {
    int *dev_values;
    size_t size = n * sizeof(int);

    cudaMalloc(&dev_values, size);
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int k, j;
    for (k = 2; k <= n; k <<= 1) {
        for (j = k >> 1; j > 0; j = j >> 1) {
            bitonicCompareAndSwapEven<<<n/k, k>>>(dev_values, j, k, n);
            cudaDeviceSynchronize();
            bitonicCompareAndSwapOdd<<<n/k, k>>>(dev_values, j, k, n);
            cudaDeviceSynchronize();
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    *sorted_values = (int*)malloc(size);
    cudaMemcpy(*sorted_values, dev_values, size, cudaMemcpyDeviceToHost);

    cudaFree(dev_values);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

__global__ void evenSortKernel(int *data, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int evenIndex = 2 * i;

    if (evenIndex < n - 1) {
        if (data[evenIndex] > data[evenIndex + 1]) {
            int temp = data[evenIndex];
            data[evenIndex] = data[evenIndex + 1];
            data[evenIndex + 1] = temp;
        }
    }
}

__global__ void oddSortKernel(int *data, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int oddIndex = 2 * i + 1;

    if (oddIndex < n - 1) {
        if (data[oddIndex] > data[oddIndex + 1]) {
            int temp = data[oddIndex];
            data[oddIndex] = data[oddIndex + 1];
            data[oddIndex + 1] = temp;
        }
    }
}

float oddEvenSort(int *values, int n, int **sorted_values) {
    int *dev_values;
    size_t size = n * sizeof(int);

    cudaMalloc(&dev_values, size);
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int threadsPerBlock = 64;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    for (int i = 0; i < n / 2; ++i) {
        evenSortKernel<<<blocks, threadsPerBlock>>>(dev_values, n);
        cudaDeviceSynchronize();
        oddSortKernel<<<blocks, threadsPerBlock>>>(dev_values, n);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    *sorted_values = (int*)malloc(size);
    cudaMemcpy(*sorted_values, dev_values, size, cudaMemcpyDeviceToHost);

    cudaFree(dev_values);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}
void printArray(const char *message, int *array, int n) {
    printf("%s\n", message);
    for (int i = 0; i < n; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

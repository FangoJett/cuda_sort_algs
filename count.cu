#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void countKernel(int* input, int* counts, int size, int range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        atomicAdd(&counts[input[i]], 1);
    }
}

__global__ void prefixSum(int* counts, int range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= range) return;

    for (int offset = 1; offset < range; offset *= 2) {
        int val = 0;
        if (i >= offset) val = counts[i - offset];
        __syncthreads();
        if (i >= offset) counts[i] += val;
        __syncthreads();
    }
}

__global__ void sortKernel(int* input, int* output, int* counts, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        int value = input[i];
        int pos = atomicSub(&counts[value], 1) - 1;
        output[pos] = value;
    }
}

void countingSort(int* input, int* output, int size, int range) {
    int* d_input, * d_output, * d_counts;

    // Alokacja i kopiowanie
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_output, size * sizeof(int));
    cudaMalloc(&d_counts, range * sizeof(int));
    cudaMemset(d_counts, 0, range * sizeof(int));

    // Ustalanie rozmiaru bloku i siatki
    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);

    // Zliczanie
    countKernel << <gridSize, blockSize >> > (d_input, d_counts, size, range);

    // Obliczanie sumy prefiksowej
    prefixSum << <1, range >> > (d_counts, range);

    // Sortowanie
    sortKernel << <gridSize, blockSize >> > (d_input, d_output, d_counts, size);

    // Kopiowanie wyników do pamięci hosta
    cudaMemcpy(output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Zwolnienie pamięci
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_counts);
}

int main() {
    const int size = 100;
    const int range = 10; // Zakres wartości (0 do range-1)
    int input[size] = { 5, 7, 3, 9, 0, 5, 8, 5, 9, 9, 1, 6, 3, 5, 7, 7, 2, 0, 5, 3, 3, 5, 2, 0, 1,
                        9, 9, 0, 6, 7, 4, 3, 0, 3, 0, 4, 8, 9, 2, 6, 9, 5, 8, 0, 5, 2, 8, 7, 0, 7,
                        2, 4, 3, 2, 9, 8, 4, 7, 9, 8, 7, 3, 7, 6, 1, 6, 4, 0, 6, 7, 6, 8, 8, 1, 5,
                        0, 3, 4, 3, 0, 4, 6, 5, 6, 6, 7, 5, 4, 3, 7, 9, 7, 5, 9, 3, 1, 4, 4, 1, 8 };
    int output[size];



    auto start = std::chrono::high_resolution_clock::now();



    countingSort(input, output, size, range);


    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;
    std::cout << "Czas wykonania: " << duration.count() << " ms\n";


    for (int i = 0; i < size; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

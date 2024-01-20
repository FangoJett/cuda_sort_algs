#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void count_digits(int* input, int* counts, int sublist_size, int num_sublists, int range) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_sublists) {
        int start = idx * sublist_size;
        for (int i = 0; i < sublist_size; i++) {
            int digit = input[start + i];
            counts[idx * range + digit]++; //atomicAdd(&counts[idx * range + digit], 1);
        }
    }
}

__global__ void sum_counts(int* counts, int* final_counts, int num_sublists, int range) {
    int digit = threadIdx.x;
    int sum = 0;
    for (int i = 0; i < num_sublists; i++) {
        sum += counts[i * range + digit];
    }
    final_counts[digit] = sum;
}


__global__ void prefixSum(int* counts, int range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= range) return;

    for (int offset = 1; offset < range; offset *= 2) {
 
        int val = 0;
        if (i >= offset) val = counts[i - offset];
     
        if (i >= offset) counts[i] += val;
    }
}


__global__ void shiftRight(int* arr, int size) {
    for (int i = size - 1; i > 0; i--) {
        arr[i] = arr[i - 1];
    }
    arr[0] = 0;
}


__global__ void sortKernel(int* input, int* output, int* counts, int size) {
    int i = threadIdx.x;
    if (i < size) {
        for (int n=0; n < size; n++) {
            if (input[n] == i) {
                output[counts[i]] = input[n];
                    counts[i]++;
          }
        }
    }
}


int main() {
    const int input_size = 100;
    const int sublist_size = 10;
    const int num_sublists = input_size / sublist_size;
    const int range = 10;
 
    int h_input[input_size] = { 5, 7, 3, 9, 0, 5, 8, 5, 9, 9, 1, 6, 3, 5, 7, 7, 2, 0, 5, 3, 3, 5, 2, 0, 1,
                                9, 9, 0, 6, 7, 4, 3, 0, 3, 0, 4, 8, 9, 2, 6, 9, 5, 8, 0, 5, 2, 8, 7, 0, 7,
                                2, 4, 3, 2, 9, 8, 4, 7, 9, 8, 7, 3, 7, 6, 1, 6, 4, 0, 6, 7, 6, 8, 8, 1, 5,
                                0, 3, 4, 3, 0, 4, 6, 5, 6, 6, 7, 5, 4, 3, 7, 9, 7, 5, 9, 3, 1, 4, 4, 1, 8 };
    std::cout << "NIEposortowana lista:\n";
    for (int i = 0; i < 100; i++) {
        std::cout << h_input[i];
    }
    std::cout << "\n\n";
    
   

    int* d_input, * d_counts, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(int));
    cudaMalloc(&d_counts, num_sublists * range * sizeof(int));
    cudaMemcpy(d_input, h_input, input_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_counts, 0, num_sublists * range * sizeof(int));
    int* d_final_counts;
    cudaMalloc(&d_final_counts, range * sizeof(int));
    int sorted[input_size];
    cudaMalloc(&d_output, input_size * sizeof(int));

    int threadsPerBlock = 256;
    int blocks = (num_sublists + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now(); 

    count_digits << <blocks, threadsPerBlock >> > (d_input, d_counts, sublist_size, num_sublists, range);
    sum_counts << <1, range >> > (d_counts, d_final_counts, num_sublists, range);
    prefixSum << <1, range >> > (d_final_counts, range);
    shiftRight << <1, 1 >> > (d_final_counts, range);    
    sortKernel << <1, range >> > (d_input, d_output, d_final_counts, input_size);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;
    std::cout << "Czas wykonania algorytmu (bez alokacji pamieci): " << duration.count() << " ms\n\n";



    cudaMemcpy(sorted, d_output, input_size * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Posortowana lista:\n" ;
    for (int i = 0; i <  100; i++) {
        std::cout << sorted[i];
    }

    std::cout << "\n\n";


    cudaFree(d_input);
    cudaFree(d_counts);
    cudaFree(d_final_counts);
    cudaFree(d_output);
    return 0;
}

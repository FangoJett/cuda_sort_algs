#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>

__global__ void count_digits(int* input, int* counts, int sublist_size, int num_sublists, int range, int digit_place) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_sublists) {
        int start = idx * sublist_size;
        for (int i = 0; i < sublist_size; i++) {
            int digit = (input[start + i] / digit_place) % 10;
            counts[idx * range + digit]++;
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


__global__ void sortKernel(int* input, int* output, int* counts, int size, int digit_place) {
    int i = threadIdx.x;
    int temp_counts;
    if (i < size) {


        int digit;
        for (int n = 0; n < size; n++) {
            digit = (input[n] / digit_place) % 10;
            if (digit == i) {
                output[counts[i]] = input[n];
                counts[i]++;
            }
        }
    }
}


int main() {
    const int input_size = 1000;
    const int sublist_size = 8;
    const int num_sublists = input_size / sublist_size;
    const int range = 10;
    const int max_number = 999;

    int h_input[input_size];


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, max_number);

    for (int i = 0; i < input_size; ++i) {
        h_input[i] = distrib(gen);
    }



    std::cout << "NIEposortowana lista:\n";
    for (int i = 0; i < input_size; i++) {
        std::cout << h_input[i] << ",";
    }
    std::cout << "\n\n";



    int* d_input, * d_counts, * d_output;
    cudaMalloc(&d_input, input_size * sizeof(int));
    cudaMalloc(&d_counts, num_sublists * range * sizeof(int));
    cudaMemcpy(d_input, h_input, input_size * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemset(d_counts, 0, num_sublists * range * sizeof(int));
    int* d_final_counts;
    cudaMalloc(&d_final_counts, range * sizeof(int));
    int sorted[input_size];
    cudaMalloc(&d_output, input_size * sizeof(int));

    int threadsPerBlock = 256;
    int blocks = (num_sublists + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now();

    int digit_place = 1;
    for (int d = 0; d < 3; d++) {  // 3 cyfry w liczbie (0-999)
        cudaMemset(d_counts, 0, num_sublists * range * sizeof(int));
        count_digits << <blocks, threadsPerBlock >> > (d_input, d_counts, sublist_size, num_sublists, range, digit_place);
        sum_counts << <1, num_sublists >> > (d_counts, d_final_counts, num_sublists, range);
        prefixSum << <1, range >> > (d_final_counts, range);
        shiftRight << <1, 1 >> > (d_final_counts, range);
        sortKernel << <1, range >> > (d_input, d_output, d_final_counts, input_size, digit_place);
        cudaMemcpy(d_input, d_output, input_size * sizeof(int), cudaMemcpyDeviceToDevice);
        digit_place *= 10;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;
    std::cout << "Czas wykonania algorytmu (bez alokacji pamieci): " << duration.count() << " ms\n\n";



    cudaMemcpy(sorted, d_output, input_size * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Posortowana lista:\n";
    for (int i = 0; i < input_size; i++) {
        std::cout << sorted[i] << ",";
    }

    std::cout << "\n\n";


    cudaFree(d_input);
    cudaFree(d_counts);
    cudaFree(d_final_counts);
    cudaFree(d_output);
    return 0;
}

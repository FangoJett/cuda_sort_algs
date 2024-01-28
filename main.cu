%%cuda
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>
#include <vector>

// Definicje funkcji kerneli
__global__ void count_digits(int* input, int* counts, int sublist_size, int num_sublists, int range, int digit_place);
__global__ void sum_counts(int* counts, int* final_counts, int num_sublists, int range);
__global__ void prefixSum(int* counts, int range);
__global__ void shiftRight(int* arr, int size);
__global__ void sortKernel(int* input, int* output, int* counts, int size, int digit_place);
__global__ void bitonicSortStep(int *dev_values, int j, int k, int n);
__global__ void bitonicCompareAndSwapEven(int *dev_values, int j, int k, int n);
__global__ void bitonicCompareAndSwapOdd(int *dev_values, int j, int k, int n);
__global__ void evenSortKernel(int *data, int n);
__global__ void oddSortKernel(int *data, int n);
__global__ void bucketSort(int* input, int* buckets, int* bucketSizes, int numBuckets, int numElements, int maxValue)
__global__ void sort_inside_BucketsKernel(int* buckets, int* bucketSizes, int numBuckets, int numElements)
__device__ void insertionSortKernel(int* bucket, int size)

// Deklaracje funkcji pomocniczych
std::pair<std::vector<int>, double> count(int* input, int input_size, int sublist_size, int num_sublists, int range, int max_number);
float bitonicSort(int *values, int n, int **sorted_values);
float bitonicSortprim(int *values, int n, int **sorted_values);
float oddEvenSort(int *values, int n, int **sorted_values);
void printArray(const char *message, int *array, int n);

int main() {
    const int input_size = 1024;
    const int max_value_input = 999;
    std::vector<int> h_input(input_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, max_value_input);

    for (int i = 0; i < input_size; ++i) {
        h_input[i] = distrib(gen);
    }

    std::cout << "Unsorted list:\n";
    for (int i = 0; i < input_size; i++) {
        std::cout << h_input[i] << ",";
    }
    std::cout << "\n\n";

    // Wykonaj sortowanie przy użyciu funkcji count
    auto [sorted, time] = count(h_input.data(), input_size, 8, input_size / 8, 10, 999);
    printArray("Sorted list using count:", sorted.data(), input_size);
    std::cout << "\nSorting time using count algorithm: " << time << " ms\n\n";
    std::cout << "\n\n";

    // Wykonaj sortowanie przy użyciu bitonicSort, bitonicSortprim i oddEvenSort
    int *sorted_values, *sorted_values1, *sorted_values2;
    float time1, time2;
    time = bitonicSortprim(h_input.data(), input_size, &sorted_values);
    printArray("Sorted list using bitonicSort in 1 kernel:", sorted_values, input_size);
    std::cout << "\n";
    std::cout << "Sorting time using BitonicSort algorithm: in 2 kernels " << time << " ms\n\n";
    std::cout << "\n\n";
    free(sorted_values);

    time1 = bitonicSort(h_input.data(), input_size, &sorted_values1);
    printArray("Sorted list using bitonicSort:", sorted_values1, input_size);
    std::cout << "\n";
    std::cout << "Sorting time using BitonicSort algorithm: in 1 kernel " << time1 << " ms\n\n";
    std::cout << "\n\n";
    free(sorted_values1);

    time2 = oddEvenSort(h_input.data(), input_size, &sorted_values2);
    printArray("Sorted list using bitonicSort:", sorted_values2, input_size);
    std::cout << "\n";
    std::cout << "Sorting time using oddEven algorithm: " << time2 << " ms\n\n";
    std::cout << "\n\n";
    free(sorted_values2);



    // Wykonaj sortowanie przy użyciu bucketsort
    //skopiowane z poprzedniego auto [sorted, time] = count(h_input.data(), input_size, 8, input_size / 8, 10, 999);
    int* sortedData = new int[input_size];
    bucketSort(h_input, sortedData, input_size, max_value_input);
    printArray("Sorted list using BucketSort:", sortedData, input_size);
    std::cout << "\nSorting time using count algorithm: " << time << " ms\n\n";
    std::cout << "\n\n";

   


    return 0;
}

// Definicje funkcji kerneli
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

__device__ void insertionSort(int* bucket, int size) {
    for (int i = 1; i < size; i++) {
        int key = bucket[i];
        int j = i - 1;
        while (j >= 0 && bucket[j] > key) {
            bucket[j + 1] = bucket[j];
            j = j - 1;
        }
        bucket[j + 1] = key;
    }
}

__global__ void bucketSortKernel(int* input, int* buckets, int* bucketSizes, int numBuckets, int numElements, int maxValue) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
        int bucketIndex = (input[idx] * numBuckets) / (maxValue + 1);
        int index = atomicAdd(&bucketSizes[bucketIndex], 1);
        buckets[bucketIndex * numElements + index] = input[idx];
    }
}

__global__ void sort_inside_BucketsKernel(int* buckets, int* bucketSizes, int numBuckets, int numElements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numBuckets) {
        int* bucket = buckets + idx * numElements;
        int size = bucketSizes[idx];
        insertionSort(bucket, size);
    }
}

//Deklaracja BucketSort
void bucketSort(int* input, int* output, int numElements, int maxValue) {
    int numBuckets = 10; // Można dostosować liczbę kubełków
    int* d_input, * d_buckets, * d_bucketSizes;

    // alokacja pamięci
    cudaMalloc(&d_input, numElements * sizeof(int));
    cudaMalloc(&d_buckets, numBuckets * numElements * sizeof(int));
    cudaMalloc(&d_bucketSizes, numBuckets * sizeof(int));

    // kopiowanie do GPU
    cudaMemcpy(d_input, input, numElements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_buckets, 0, numBuckets * numElements * sizeof(int));
    cudaMemset(d_bucketSizes, 0, numBuckets * sizeof(int));

    // rozdział do kubłów
    int blockSize = 256;
    int numBlocks = (numElements + blockSize - 1) / blockSize;
    bucketSortKernel<<<numBlocks, blockSize>>>(d_input, d_buckets, d_bucketSizes, numBuckets, numElements, maxValue);

    // sorotwanie pojedyńczych kubłów
    int bucketsBlockSize = 256;
    int bucketsNumBlocks = (numBuckets + bucketsBlockSize - 1) / bucketsBlockSize;
    sort_inside_BucketsKernel<<<bucketsNumBlocks, bucketsBlockSize>>>(d_buckets, d_bucketSizes, numBuckets, numElements);

    // zbieranie posorotowanych danych
    for (int i = 0, k = 0; i < numBuckets; ++i) {
        int size;
        cudaMemcpy(&size, &d_bucketSizes[i], sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(output + k, d_buckets + i * numElements, size * sizeof(int), cudaMemcpyDeviceToHost);
        k += size;
    }

    cudaFree(d_input);
    cudaFree(d_buckets);
    cudaFree(d_bucketSizes);
}

// Deklaracja funkcji count
std::pair<std::vector<int>, double> count(int* input, int input_size, int sublist_size, int num_sublists, int range, int max_number) {
    int *d_input, *d_counts, *d_output, *d_final_counts;
    std::vector<int> sorted(input_size);

    cudaMalloc(&d_input, input_size * sizeof(int));
    cudaMalloc(&d_counts, num_sublists * range * sizeof(int));
    cudaMemcpy(d_input, input, input_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_final_counts, range * sizeof(int));
    cudaMalloc(&d_output, input_size * sizeof(int));

    int threadsPerBlock = 256;
    int blocks = (num_sublists + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now();

    int digit_place = 1;
    for (int d = 0; d < 3; d++) {  // 3 cyfry w liczbie (0-999)
        cudaMemset(d_counts, 0, num_sublists * range * sizeof(int));
        count_digits<<<blocks, threadsPerBlock>>>(d_input, d_counts, sublist_size, num_sublists, range, digit_place);
        sum_counts<<<1, range>>>(d_counts, d_final_counts, num_sublists, range);
        prefixSum<<<1, range>>>(d_final_counts, range);
        shiftRight<<<1, 1>>>(d_final_counts, range);
        sortKernel<<<1, range>>>(d_input, d_output, d_final_counts, input_size, digit_place);
        cudaMemcpy(d_input, d_output, input_size * sizeof(int), cudaMemcpyDeviceToDevice);
        digit_place *= 10;
    }

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;
    double time = duration.count();

    cudaMemcpy(sorted.data(), d_output, input_size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_counts);
    cudaFree(d_final_counts);
    cudaFree(d_output);

    return {sorted, time};
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

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// Funkcja do znalezienia maksymalnej wartości w wektorze
int getMax(const vector<int>& arr) {
    cout<<(*max_element(arr.begin(), arr.end()));
    return *max_element(arr.begin(), arr.end());
}

// Funkcja do wykonania sortowania przez zliczanie dla arr[]
// według cyfry reprezentowanej przez exp.
void countSort(vector<int>& arr, int exp) {
    int n = arr.size();
    vector<int> output(n);
    int count[10] = {0};

    // Zapisz liczbę wystąpień w count[]
    for (int i = 0; i < n; i++) {
        count[(arr[i] / exp) % 10]++;
    }

    // Zmień count[i] aby zawierał rzeczywistą pozycję cyfry w output[]
    for (int i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }

    // Zbuduj wektor wyjściowy
    for (int i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }

    // Skopiuj wektor wyjściowy do arr[], aby arr[] zawierał posortowane liczby
    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }
}

// Funkcja do sortowania arr[] według algorytmu Radix Sort
void radixSort(vector<int>& arr) {
    // Znajdź maksymalną liczbę, aby poznać liczbę cyfr
    int m = getMax(arr);

    // Wykonaj sortowanie przez zliczanie dla każdej cyfry. Zauważ, że zamiast
    // przechodzić cyfrę po cyfrze, używamy exp. exp jest 10^i, gdzie i jest aktualną cyfrą.
    for (int exp = 1; m / exp > 0; exp *= 10) {
        countSort(arr, exp);
    }
}

int main() {
    vector<int> arr = {170, 45, 75, 90, 802, 24, 2, 66};
    radixSort(arr);

    cout << "Posortowany wektor: ";
    for (int i = 0; i < arr.size(); i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    return 0;
}
#include <iostream>
#include <array>
#include <immintrin.h>
#include <chrono>
#include "MockNet.h"
#include <iostream>
#include <assert.h>


void bench_primitive()
{
	constexpr int samples = 10000;
	std::array<uint8_t, 1024> src1 = {};
	std::array<int8_t, 1024> src2 = {};
	std::array<int32_t, 256> src3 = {};
	std::array<int32_t, 256> volatile dst1 = {};
	for (size_t i = 0; i < 1024; i++) {
		src1[i] = static_cast<uint8_t>(i % 240);
		src2[i] = static_cast<int8_t>(i % 100);
	}

	for (size_t i = 0; i < 256; i++) {
		src3[i] = static_cast<int32_t>(i);
	}
	int64_t total_time = 0;
	int64_t total_sum = 0;
	//Loop 1000 times to get a good average
	for (int j = 0;j < samples;j++) {
		auto start = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < 256; i++) {
			dst1[i] =
				static_cast<int32_t>(src1[i * 4]) * static_cast<int32_t>(src2[i * 4]) +
				static_cast<int32_t>(src1[i * 4 + 1]) * static_cast<int32_t>(src2[i * 4 + 1]) +
				static_cast<int32_t>(src1[i * 4 + 2]) * static_cast<int32_t>(src2[i * 4 + 2]) +
				static_cast<int32_t>(src1[i * 4 + 3]) * static_cast<int32_t>(src2[i * 4 + 3]) +
				static_cast<int32_t>(src3[i]);
			total_sum += dst1[i];
		}
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		total_time += duration;
		total_sum = 0;
	}

	std::cout << "Autovec code took an average of " << total_time / samples << " nanoseconds" << std::endl;
	std::cout << "Primitive autovec total sum is: " << total_sum << std::endl;
	std::cout << " \n--- \n";
#if defined(__AVX512VNNI__) && defined(__AVX512F__)
	std::cout << "Now trying AVX512VNNI instrisics\n";
	__m512i volatile _src1, _src2, _src3, _dst;
	constexpr int register_size = 512;
	constexpr int int8_per_register = 512 / 8;
	constexpr int int32_per_register = 512 / 32;
	int64_t final_sum = 0;
	total_time = 0;
	//Loop 1000 times to get a good average
	for (int j = 0;j < samples;j++) {
		auto simd_start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1024 / int8_per_register;i++) {
			_src1 = _mm512_loadu_si512((__m512i const*)&src1[int8_per_register * i]);
			_src2 = _mm512_loadu_si512((__m512i const*)&src2[int8_per_register * i]);
			_src3 = _mm512_loadu_si512((__m512i const*)&src3[int32_per_register * i]);
			_dst = _mm512_dpbusd_epi32(_src3, _src1, _src2);
			final_sum += _mm512_reduce_add_epi32(_dst);
		}
		auto simd_stop = std::chrono::high_resolution_clock::now();
		auto simd_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(simd_stop - simd_start).count();
		total_time += simd_duration;
	}

	std::cout << "SIMD code took an average of " << total_time / samples << " nanoseconds" << std::endl;
	std::cout << "Primitive SIMD total sum is: " << final_sum << std::endl;

#else
	std::cout << " AVX512VNNI Support not found\n";
#endif
}


void bench_nnue()
{

	MockNet net = MockNet();
	net.init();
	net.move(net.board_accumulator, 6, 10, 25);
	auto output = net.output(net.board_accumulator);
	std::cout << "Autovec output is: " << output << std::endl;
#if defined(__AVX512VNNI__) && defined(__AVX512F__)
	auto outputSIMD = net.outputSIMD(net.board_accumulator);
	std::cout << "SIMD output is: " << outputSIMD << std::endl;
#endif

}




//Convolute an input matrix with a kernel and add a bias
/* array([[28., 31., 34.],
	   [40., 43., 46.],
	   [52., 55., 58.]])*/
void convolute(std::array<std::array<int8_t, 4>, 4> input, std::array<std::array<int8_t, 2>, 2> kernel, int32_t bias)
{
	constexpr int samples = 10000;
	int64_t total_time = 0;
	int volatile convolute = 0; // This holds the convolution results for an index.
	int x, y; // Used for input matrix index
	constexpr int input_rows = 4, input_columns = 4, kernel_rows = 2, kernel_columns = 2;

	//Size output array
	const int width = (input_columns - kernel_columns) + 1;

	const int height = (input_rows - kernel_rows) + 1;

	std::array<std::array<int32_t, height>, width> output;
	for (int j = 0;j < samples;j++) {
		auto start = std::chrono::high_resolution_clock::now();
		// Going over every row of the input
		for (int i = 0; i < input_rows; i++)
		{
			// Going over every column of each row
			for (int j = 0; j < input_columns; j++)
			{
				//Pinpot input value we are working on
				x = i;
				y = j;
				//Quick check for if we are out of bounds
				if (!(x + kernel_rows <= input_rows)) break;
				if (!(y + kernel_columns <= input_columns)) break;

				// Going over every row of the input
				for (int k = 0; k < kernel_rows; k++)
				{
					// Going over every column of each row
					for (int l = 0; l < kernel_columns; l++)
					{
						// Convolute input square with kernel square
						convolute += input[x][y] * kernel[k][l];
						y++; // Move right.
					}
					x++; // Move down.
					y = j; // Restart column position
				}
				assert(i < width && j < height);
				output[i][j] = convolute + bias; // Add result to output matrix.
				convolute = 0; // Needed before we move on to the next index.
			}
		}
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		total_time += duration;
	}
	std::cout << "Autovec convolution took: " << total_time / samples << std::endl;
	for (std::array<int, width> row : output) {
		for (int element : row) {
			std::cout << element << " ";
		}
		std::cout << std::endl;
	}
	return;
}


//Convolute an input matrix with a kernel and add a bias

void convoluteSIMD(std::array<std::array<int8_t, 4>, 4> input, std::array<std::array<int8_t, 2>, 2> kernel, int32_t bias)
{
#if defined(__AVX512VNNI__) && defined(__AVX512F__)
	std::cout << "Now trying AVX512VNNI instrisics\n";
	__m512i volatile _src1, _src2, _src3, _dst;
	int convolutions[512];
	int8_t src1[512];
	int8_t src2[512];
	constexpr int samples = 10000;
	int64_t total_time = 0;
	auto convolutions_totals=0;
	constexpr int input_rows = 4, input_columns = 4, kernel_rows = 2, kernel_columns = 2;

	//Size output array
	const int width = (input_columns - kernel_columns) + 1;

	const int height = (input_rows - kernel_rows) + 1;

	// Load the bias vector
	_src3 = _mm512_set1_epi32(bias);
	//Duplicate elements in the kernel vector 
	_src2 = _mm512_set1_epi32((kernel[0][0]) + (kernel[0][1] << 8) + ((kernel[1][0] + 1) << 16) + ((kernel[1][1] + 1) << 24));
	//Load the input vector in the exact order we need it in
	_src1 = _mm512_set_epi32(
		0, 0, 0, 0, 0, 0, 0,
		input[2][2] + (input[2][3] << 8) + (input[3][2] << 16) + (input[3][3] << 24),
		input[2][1] + (input[2][2] << 8) + (input[3][1] << 16) + (input[3][2] << 24),
		input[2][0] + (input[2][1] << 8) + (input[3][0] << 16) + (input[3][1] << 24),
		input[1][2] + (input[1][3] << 8) + (input[2][2] << 16) + (input[2][3] << 24),
		input[1][1] + (input[1][2] << 8) + (input[2][1] << 16) + (input[2][2] << 24),
		input[1][0] + (input[1][1] << 8) + (input[2][0] << 16) + (input[2][1] << 24),
		input[0][2] + (input[0][3] << 8) + (input[1][2] << 16) + (input[1][3] << 24),
		input[0][1] + (input[0][2] << 8) + (input[1][1] << 16) + (input[1][2] << 24),
		input[0][0] + (input[0][1] << 8) + (input[1][0] << 16) + (input[1][1] << 24)
	);
	for (int j = 0;j < samples;j++) {
		auto start = std::chrono::high_resolution_clock::now();
		_dst = _mm512_dpbusd_epi32(_src3, _src1, _src2);

		_mm512_storeu_epi32(convolutions, _dst);
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
		total_time += duration;
		convolutions_totals += convolutions[0];
	}
	for (int i = 0; i < width * height;i++) {

		std::cout << convolutions[i] << " ";
		if ((i + 1) % width == 0) std::cout << std::endl;
}
#endif
}


void bench_conv() {

	std::array<std::array<int8_t, 4>, 4> A = { 1, 2, 3, 4,5, 6, 7, 8,9, 10, 11, 12,13, 14, 15, 16 };
	std::array<std::array<int8_t, 2>, 2> Kernel = { 0,-1 ,
												   -1, 5 };

	convolute(A, Kernel, 5);
	convoluteSIMD(A, Kernel, 5);
	return;
}

int main()
{
	bench_primitive();
	bench_nnue();
	bench_conv();
	return 0;
}
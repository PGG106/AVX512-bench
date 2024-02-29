#include "MockNet.h"
#include <chrono>
#include <chrono>
#include <iostream>
void MockNet::init()
{
	for (int j = 0;j < INPUT_WEIGHTS - 1;j++) {
		for (int k = 0;k < HIDDEN_SIZE - 1;k++) {
			featureWeights[INPUT_WEIGHTS * HIDDEN_SIZE] = rand();
		}
	}


	for (int i = 0;i < HIDDEN_SIZE;i++) {
		featureBias[i] = rand();
		outputWeights[i] = outputWeights[i * 2] = rand();
	}
	outputBias = rand();
}
//Simulates an update of the net
void MockNet::move(MockNet::accumulator& board_accumulator, int piece, int from, int to)
{
	auto Fromindex = from + piece * 64;
	auto Toindex = to + piece * 64;

	auto whiteSub = &featureWeights[Fromindex * 512];
	auto whiteAdd = &featureWeights[Toindex * 512];
	for (int i = 0; i < HIDDEN_SIZE; i++)
	{
		board_accumulator[i] = board_accumulator[i] - whiteSub[i] + whiteAdd[i];
	}
}

//Simulates an update of the net
int32_t MockNet::output(const MockNet::accumulator& board_accumulator)
{
	int32_t output = 0;
	auto start = std::chrono::high_resolution_clock::now();
	for (int j = 0;j < 1500;j++) {
		for (int i = 0; i < HIDDEN_SIZE; i++)
		{
			output += board_accumulator.data()[i] * static_cast<int32_t>(outputWeights[i]);
		}
	}
	int32_t unsquared = output / 255 + outputBias;
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	std::cout <<"Net inference with autovec took "<< duration / 1500 << std::endl;
	return unsquared * 400 / (64 * 255);
}

//Simulates an update of the net
int32_t MockNet::outputSIMD(const MockNet::accumulator& board_accumulator)
{
#if defined(__AVX512VNNI__) && defined(__AVX512F__)
	constexpr int register_size = 512;
	constexpr int int8_per_register = 512 / 8;
	constexpr int int32_per_register = 512 / 32;
	int32_t output = 0;
	__m512i _src1, _src2, _src3, _dst;
	auto start = std::chrono::high_resolution_clock::now();
	for (int j = 0;j < 1500;j++) {

		for (int i = 0; i < HIDDEN_SIZE / int8_per_register;i++) {
			_src1 = reg_loadu((__m512i const *)&board_accumulator.data()[0]);
			_src2 = reg_loadu((__m512i const *)&outputWeights[0]);
			_src3 = _mm512_set1_epi8(1);
			_dst = dpbusd(_src3, _src1, _src2);
			output += reduce_add(_dst);
		}
	
	}
	int32_t unsquared = output / 255 + outputBias;
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	std::cout <<"Net inference with SIMD took "<< duration / 1500 << std::endl;
	return unsquared * 400 / (64 * 255);
#endif
	return 0;
}
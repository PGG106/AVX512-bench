#pragma once
#include <cstdint>
#include <array>
#include <immintrin.h>
#include "simd.h"

constexpr int INPUT_WEIGHTS = 768;
constexpr int HIDDEN_SIZE = 512;
class MockNet {
public:
	using accumulator = std::array<int8_t, 1024>;
	void init();
	void move(MockNet::accumulator& board_accumulator, int piece, int from, int to);
	int32_t output(const MockNet::accumulator& board_accumulator);
	int32_t outputSIMD(const MockNet::accumulator& board_accumulator);
	int16_t featureWeights[INPUT_WEIGHTS * HIDDEN_SIZE];
	int16_t featureBias[HIDDEN_SIZE];
	int16_t outputWeights[HIDDEN_SIZE * 2];
	int16_t outputBias;
	accumulator board_accumulator;

};

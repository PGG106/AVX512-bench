#include <iostream>
#include <array>
#include <immintrin.h>
#include <chrono>

template<typename T>
void print_register(const __m512i& zmm0)
{
	constexpr auto entries_count = 512 / (8 * sizeof(T));

	std::array<T, entries_count> values = {};
	_mm512_storeu_si512((__m512i*)values.data(), zmm0);

	for (size_t i = 0; i < entries_count; i++) {
		std::cout << static_cast<int64_t>(values[i]) << "\n";
	}
}


int main()
{
	std::array<uint8_t, 1024> src1 = {};
	std::array<int8_t, 1024> src2 = {};
	std::array<int32_t, 256> src3 = {};
	std::array<int32_t, 256> dst1 = {};

	for (size_t i = 0; i < 1024; i++) {
		src1[i] = static_cast<uint8_t>(i % 240);
		src2[i] = static_cast<int8_t>(i % 100);
	}

	for (size_t i = 0; i < 256; i++) {
		src3[i] = static_cast<int32_t>(i);
	}
	auto start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < 256; i++) {
		dst1[i] = static_cast<int32_t>(
			(src1[i * 4] * src2[i * 4]) +
			(src1[i * 4 + 1] * src2[i * 4 + 1]) +
			(src1[i * 4 + 2] * src2[i * 4 + 2]) +
			(src1[i * 4 + 3] * src2[i * 4 + 3]) +
			src3[i]
			);
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
	std::cout << "Autovec code took " << duration << " nanoseconds" << std::endl;
	for (size_t i = 240; i < 256; i++) {
		std::cout << static_cast<int64_t>(dst1[i]) << "\n";
	}

	std::cout << " --- \n";
#if defined(__AVX512VNNI__) && defined(__AVX512F__)
	std::cout << "Now trying AVX512VNNI instrisics\n";
	__m512i _src1, _src2, _src3, _dst;
	auto simd_start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < (1024 / 64);i++) {
		_src1 = _mm512_loadu_si512((__m512i const*)&src1[64 * i]);
		_src2 = _mm512_loadu_si512((__m512i const*)&src2[64 * i]);
		_src3 = _mm512_loadu_si512((__m512i const*)&src3[16 * i]);
		_dst = _mm512_dpbusd_epi32(_src3, _src1, _src2);
	}
	auto simd_stop = std::chrono::high_resolution_clock::now();
	auto simd_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(simd_stop - simd_start).count();
	std::cout << "SIMD code took " << simd_duration << " nanoseconds" << std::endl;
	print_register<int32_t>(_dst);
#else
	std::cout << " AVX512VNNI Support not found\n";
#endif
	return 0;
}
#include <cstdint>
#include <iostream>
#include <array>
#include <immintrin.h>
#include <cstring>

template <typename T>
void print_register(__m512i r) {
	constexpr auto entries_count = 512 / (8 * sizeof(T));
	std::array<T, entries_count> values = {};
	std::cout << "entries count is: " << entries_count << "\n";
	_mm512_storeu_epi32(values.data(), r);
	std::cout << +values[0] << std::endl;

}
int main()
{
	alignas(64) uint8_t src1[1024] = {};
	alignas(64) int8_t src2[1024] = {};
	alignas(64) int32_t src3[256] = {};
	alignas(64) int32_t dst[256] = {};

	for (int i = 0;i < 1024;i++) {
		src1[i] = i % 240;
		src2[i] = i % 100;
	}

	for (int i = 0;i < 256;i++) {
		src3[i] = i;
	}

#if defined(__AVX512VNNI__) && defined(__AVX512F__)
	std::cout << "Now trying AVX512VNNI instrisics\n";
	__m512i _src1, _src2, _src3, _dst;
	//Load data in the arrays
	_src1 = _mm512_loadu_epi8(src1);
	_src2 = _mm512_loadu_epi8(src2);
	_src3 = _mm512_loadu_epi32(src3);

	_dst = _mm512_dpbusd_epi32(_src3, _src1, _src2);
	print_register<std::int8_t>(_src1);

	print_register<std::int8_t>(_src2);

	print_register<std::int32_t>(_src3);

	print_register<std::int32_t>(_dst);

#else
	std::cout << " AVX512VNNI Support not found\n";
#endif
	return 0;
}


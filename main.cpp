#include <cstdint>
#include <chrono>
#include <iostream>
#include <array>
#include <immintrin.h>
#include <cstring>
/*AVX512 - VNNI INTRODUCES 4 NEW INSTRUCTIONS
* VPDPBUSD, VPDPBUSDS, VPDPWSSD, VPDPWSSDS
* VPDPBUSD AND VPDPWSSD WORK IDENTICALLY TO VPDPBUSDS AND VPDPWSSDS BUT THE LATTER ARE SATURETED
* FOR BENCHMARKING PURPOSES WE CAN FOCUS ON JUST VPDPBUSD AND VPDPWSSD
*
* VPDPBUSD - Multiplies the individual bytes (8-bit) of the first source operand by the corresponding bytes (8-bit) of the second source operand,
producing intermediate word (16-bit) results which are summed and accumulated in the double word (32-bit) of the destination operand
*
* Multiplies the individual words (16-bit) of the first source operand by the corresponding word (16-bit) of the second source operand,
producing intermediate word results which are summed and accumulated in the double word (32-bit) of the destination operand
*
*
*
*/
template <typename T>
void print_register(__m512i r) {
	constexpr auto entries_count1 = 512 / (8 * sizeof(T));
	std::array<T, entries_count1> values = {};
	std::cout << "entries count is: " << entries_count1 << "\n";
	_mm512_storeu_epi32(values.data(), r);
	std::cout << +values[0] << std::endl;

}
int main()
{
	/*
	VPDPBUSD offers 3 intrisics for 512 bits wide registers:
	_mm512_dpbusd_epi32
	_mm512_mask_dpbusd_epi32
	_mm512_maskz_dpbusd_epi32

	the main action all 3 instrisics do is:
	Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in a with corresponding signed 8-bit integers in b, producing 4 intermediate signed 16-bit results.
	Sum these 4 results with the corresponding 32-bit integer in src, and store the packed 32-bit results in dst.

	_mm512_mask_dpbusd_epi32 and _mm512_maskz_dpbusd_epi32 have additional uses of a write and zero mask respectively we don't care about from a performance standpoint

	*/

	//We start off by creating the arrays to manipulate with the istructions
	uint8_t src1[1024];
	uint8_t src2[1024];
	int32_t src3[256];
	int32_t dst[256];

	for (int i = 0;i < 1024;i++) {
		src1[i] = i % 240;
		src2[i] = i % 240;
	}
	for (int i = 0;i < 256;i++) {
		src3[i] = i;
	}
	//Mimick the desired end result without using instrisics
	for (int i = 0;i < 256;i++) {
		dst[i] = (src1[i * 4] * src2[i * 4]) +
			(src1[i * 4 + 1] * src2[i * 4 + 1]) +
			(src1[i * 4 + 2] * src2[i * 4 + 2]) +
			(src1[i * 4 + 3] * src2[i * 4 + 3]) +
			src3[i];
	}
	/**************************************** AVX512VNNI instrisics ****************************/
#if defined(__AVX512VNNI__) && defined(__AVX512F__)
	std::cout << "Now trying AVX512VNNI instrisics\n";
	//The goal is doing dst = (src*src2) + src3 with 1 instruction
	//Create the registers to load the arrays into
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


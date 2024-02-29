#pragma once
#pragma once
#include <immintrin.h>
//Wrap the required set of intrisics into something a bit nicer
#if defined(__AVX512F__)
#define reg_loadu   _mm512_load_si512
#define reg_store   _mm512_store_epi32
#define dpbusd      _mm512_dpbusd_epi32
#define reduce_add  _mm512_reduce_add_epi32
#endif
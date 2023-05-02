#include <cstdint>
#include <iostream>
#include <array>
#include <immintrin.h>

template <typename T>
auto print_register(const __m512i r) -> void {
    constexpr auto entries_count = 512 / (8 * sizeof(T));
    std::array<T, entries_count> values;
    _mm512_storeu_epi32(values.data(), r);
    std::cout << static_cast<int>(values[0]) << std::endl;
}

auto main() -> int {
    std::array<std::uint8_t, 1024> src1 = {};
    const auto _src1 = _mm512_loadu_epi8(src1.data());
    print_register<std::int8_t>(_src1);
    return 0;
}

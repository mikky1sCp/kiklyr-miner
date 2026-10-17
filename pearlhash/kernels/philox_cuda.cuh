// Philox-4x32-10 + rejection sampling для pearlhash (bit-exact с NumPy-эталоном).
//
// Константы — Random123 (Salmon et al., SC '11) / curand Philox-4x32-10:
//   M0 = 0xD2511F53, M1 = 0xCD9E8D57, W0 = 0x9E3779B9, W1 = 0xBB67AE85
// Раунд (каноническая раскладка curand/Random123):
//   (hi0, lo0) = mulu64(x0, M0);  (hi1, lo1) = mulu64(x2, M1)
//   x0' = hi1 ^ x1 ^ k0
//   x1' = lo1 ^ x3 ^ k1
//   x2' = hi0 ^ x0 ^ W0
//   x3' = lo0 ^ x0 ^ W1   (все выражения справа — СТАРЫЕ значения)
//
// Layout слов совпадает с pearlhash/philox.py: блоки счётчика idx идут
// подряд, слова word = block*4 + lane (r0,r1,r2,r3). Rejection sampling
// воспроизводит PhiloxRng.randint_range побитово (тот же limit и порядок).

#pragma once
#include <cstdint>

#define PRL_PHILOX_ROUNDS 10

__device__ __forceinline__ void prl_philox_round(uint32_t &x0, uint32_t &x1,
                                                 uint32_t &x2, uint32_t &x3,
                                                 const uint32_t k0,
                                                 const uint32_t k1) {
    const uint64_t p0 = (uint64_t)x0 * (uint64_t)0xD2511F53ull;
    const uint64_t p1 = (uint64_t)x2 * (uint64_t)0xCD9E8D57ull;
    const uint32_t hi0 = (uint32_t)(p0 >> 32), lo0 = (uint32_t)p0;
    const uint32_t hi1 = (uint32_t)(p1 >> 32), lo1 = (uint32_t)p1;
    const uint32_t n0 = hi1 ^ x1 ^ k0;
    const uint32_t n1 = lo1 ^ x3 ^ k1;
    const uint32_t n2 = hi0 ^ x0 ^ 0x9E3779B9u;
    const uint32_t n3 = lo0 ^ x0 ^ 0xBB67AE85u;
    x0 = n0; x1 = n1; x2 = n2; x3 = n3;
}

// Один блок Philox: counter (c0..c3) + key (k0,k1) -> 4 слова r0..r3.
__device__ __forceinline__ void prl_philox4(uint32_t c[4], const uint32_t k0,
                                            const uint32_t k1) {
#pragma unroll
    for (int r = 0; r < PRL_PHILOX_ROUNDS; ++r)
        prl_philox_round(c[0], c[1], c[2], c[3], k0, k1);
}

// Слово № `word` потока (key = seed{lo, hi+salt}).
__device__ __forceinline__ uint32_t prl_philox_word(uint64_t word,
                                                    uint32_t k0, uint32_t k1) {
    const uint64_t block = word >> 2;
    const int lane = (int)(word & 3);
    uint32_t c[4] = {(uint32_t)block, (uint32_t)(block >> 32), 0u, 0u};
    prl_philox4(c, k0, k1);
    return c[lane];
}

// Ключ роли: salt сдвигает старшее слово ключа (как в philox.py).
__device__ __forceinline__ void prl_role_key(uint64_t seed, int salt,
                                             uint32_t &k0, uint32_t &k1) {
    k0 = (uint32_t)(seed & 0xFFFFFFFFull);
    k1 = (uint32_t)(((seed >> 32) + (uint64_t)salt) & 0xFFFFFFFFull);
}

// Равномерное int32 в [lo, hi] через rejection sampling; вызывается
// ПОСЛЕДОВАТЕЛЬНО по индексу слова i = 0..count-1 одним потоком —
// воспроизводит алгоритм randint_range эталона без modulo bias.
__device__ __forceinline__ int32_t prl_reject_next(
    uint32_t k0, uint32_t k1, uint64_t *consumed_words, int32_t lo,
    int32_t hi) {
    const int64_t span = (int64_t)hi - (int64_t)lo + 1;
    const uint32_t limit =
        (uint32_t)((1ull << 32) - ((1ull << 32) % (uint64_t)span));
    for (;;) {
        const uint32_t w = prl_philox_word((*consumed_words)++, k0, k1);
        if (w < limit) return (int32_t)((uint64_t)w % (uint64_t)span) + lo;
    }
}

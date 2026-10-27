// BLAKE3 (spec reference, 7-раундовый Feistel) для pearlhash.
// Поддерживает:
//   * одноблочное хеширование <=1024 байт (hash_grid: fold 32B, тайлы 8KB);
//   * tree-режим с chunk/parent флагами и counter-корнем для длинных
//     входов (>1024B) — побитово эквивалентно python blake3.
// Константы/инициализация/перестановки — из BLAKE3 spec (oconnor/blake3 ref).

#pragma once
#include <cstdint>

#define PRL_B3_IV0 0x6A09E667u
#define PRL_B3_IV1 0xBB67AE85u
#define PRL_B3_IV2 0x3C6EF372u
#define PRL_B3_IV3 0xA54FF53Au
#define PRL_B3_IV4 0x510E527Fu
#define PRL_B3_IV5 0x9B05688Cu
#define PRL_B3_IV6 0x1F83D9ABu
#define PRL_B3_IV7 0x5BE0CD19u

#define PRL_B3_CHUNK_START  0x01u
#define PRL_B3_CHUNK_END    0x02u
#define PRL_B3_PARENT_START 0x04u
#define PRL_B3_PARENT_END   0x08u
#define PRL_B3_ROOT         0x10u
#define PRL_B3_KEYED_HASH   0x20u
#define PRL_B3_DERIVE_KEY   0x40u

__device__ __constant__ static const uint32_t PRL_B3_MSG_PERM[16] = {
    2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
__device__ __constant__ static const uint32_t PRL_B3_OUT_LEN[6] = {
    8, 9, 10, 11, 12, 13};

__device__ __forceinline__ uint32_t prl_b3_rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ void prl_b3_g(uint32_t s[16], int a, int b, int c,
                                         int d, uint32_t mx, uint32_t my) {
    s[a] = s[a] + s[b] + mx;
    s[d] = prl_b3_rotr32(s[d] ^ s[a], 16);
    s[c] = s[c] + s[d];
    s[b] = prl_b3_rotr32(s[b] ^ s[c], 12);
    s[a] = s[a] + s[b] + my;
    s[d] = prl_b3_rotr32(s[d] ^ s[a], 8);
    s[c] = s[c] + s[d];
    s[b] = prl_b3_rotr32(s[b] ^ s[c], 7);
}

__device__ __forceinline__ void prl_b3_round(uint32_t s[16],
                                             const uint32_t m[16]) {
    prl_b3_g(s, 0, 4, 8, 12, m[0], m[1]);
    prl_b3_g(s, 1, 5, 9, 13, m[2], m[3]);
    prl_b3_g(s, 2, 6, 10, 14, m[4], m[5]);
    prl_b3_g(s, 3, 7, 11, 15, m[6], m[7]);
    prl_b3_g(s, 0, 5, 10, 15, m[8], m[9]);
    prl_b3_g(s, 1, 6, 11, 12, m[10], m[11]);
    prl_b3_g(s, 2, 7, 8, 13, m[12], m[13]);
    prl_b3_g(s, 3, 4, 9, 14, m[14], m[15]);
}

__device__ __forceinline__ void prl_b3_permute(uint32_t m[16]) {
    const uint32_t t0 = m[0], t1 = m[1], t2 = m[2], t3 = m[3];
    m[0] = m[7];   m[1] = m[4];  m[2] = m[13]; m[3] = m[1];
    m[4] = m[10];  m[5] = m[6];  m[6] = m[15]; m[7] = m[3];
    m[8] = m[12];  m[9] = m[0];  m[10] = m[9]; m[11] = m[2];
    m[12] = m[14]; m[13] = m[5]; m[14] = m[11]; m[15] = m[8];
    (void)t0; (void)t1; (void)t2; (void)t3;
}

// Состояние chaining value (8 слов), как Input in spec reference.
struct PrlB3Input {
    uint32_t cv[8];
    uint64_t counter;
    uint32_t block_len;
    uint32_t flags;
};

__device__ __forceinline__ PrlB3Input prl_b3_new_input(const uint32_t key[8],
                                                       uint64_t counter,
                                                       uint32_t block_len,
                                                       uint32_t flags) {
    PrlB3Input in;
    for (int i = 0; i < 8; ++i) in.cv[i] = key[i];
    in.counter = counter;
    in.block_len = block_len;
    in.flags = flags;
    return in;
}

// Полный выход компрессии: out16[0..7]=low^(high), out16[8..15]=high^input_cv.
__device__ __forceinline__ void prl_b3_compress_full(const PrlB3Input &in,
                                                    const uint32_t block[16],
                                                    uint32_t out16[16]) {
    uint32_t s[16];
    uint32_t m[16];
    for (int i = 0; i < 16; ++i) m[i] = block[i];
    for (int i = 0; i < 8; ++i) s[i] = in.cv[i];
    s[8] = PRL_B3_IV0; s[9] = PRL_B3_IV1; s[10] = PRL_B3_IV2;
    s[11] = PRL_B3_IV3; s[12] = (uint32_t)in.counter;
    s[13] = (uint32_t)(in.counter >> 32);
    s[14] = in.block_len; s[15] = in.flags;
#pragma unroll
    for (int r = 0; r < 7; ++r) {
        prl_b3_round(s, m);
        if (r != 6) prl_b3_permute(m);
    }
    for (int i = 0; i < 8; ++i) out16[i] = s[i] ^ s[i + 8];
    for (int i = 0; i < 8; ++i) out16[i + 8] = s[i + 8] ^ in.cv[i];
}

__device__ __forceinline__ void prl_b3_extend_cv(const PrlB3Input &in,
                                                 const uint32_t block[16],
                                                 uint32_t cv_out[8]) {
    uint32_t o16[16];
    prl_b3_compress_full(in, block, o16);
    for (int i = 0; i < 8; ++i) cv_out[i] = o16[i];
}

// Чтение little-endian uint32 из байтов.
__device__ __forceinline__ uint32_t prl_b3_load32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

// IV = "ключ" для unkeyed hash (blake3.blake3(data)).
__device__ __forceinline__ void prl_b3_empty_key(uint32_t out[8]) {
    out[0] = PRL_B3_IV0; out[1] = PRL_B3_IV1; out[2] = PRL_B3_IV2;
    out[3] = PRL_B3_IV3; out[4] = PRL_B3_IV4; out[5] = PRL_B3_IV5;
    out[6] = PRL_B3_IV6; out[7] = PRL_B3_IV7;
}

__device__ __forceinline__ void prl_b3_parse_block(const uint8_t *bytes64,
                                                   uint32_t out[16]) {
    for (int i = 0; i < 16; ++i) out[i] = prl_b3_load32(bytes64 + 4 * i);
}

// Хеш <=1024 байт одним чанком (single-chunk root), результат 32 байта.
// Эквивалентен python blake3(data).digest() при len(data) <= 1024.
__device__ __forceinline__ void prl_b3_hash_small(const uint8_t *data,
                                                  uint32_t len,
                                                  const uint32_t key_words[8],
                                                  uint32_t kind_flags,
                                                  uint8_t out32[32]) {
    const uint32_t blocks = (len == 0) ? 1 : (len + 63) / 64;
    uint32_t cv[8];
    for (int i = 0; i < 8; ++i) cv[i] = key_words[i];
    for (uint32_t b = 0; b + 1 < blocks; ++b) {
        PrlB3Input in = prl_b3_new_input(
            cv, b, 64,
            (b == 0) ? (PRL_B3_CHUNK_START | kind_flags) : kind_flags);
        uint32_t blk[16];
        prl_b3_parse_block(data + 64 * b, blk);
        prl_b3_extend_cv(in, blk, cv);
    }
    {
        const uint32_t last = blocks - 1;
        const uint32_t off = 64 * last;
        const uint32_t rem = len - off;  // 0 только если len кратно 64 и >0
        PrlB3Input in = prl_b3_new_input(
            cv, last, (blocks == 1) ? len : (rem == 0 ? 64 : rem),
            PRL_B3_CHUNK_START | PRL_B3_CHUNK_END | PRL_B3_ROOT | kind_flags);
        uint8_t buf[64];
        for (int i = 0; i < 64; ++i) {
            const uint32_t idx = off + i;
            buf[i] = (idx < len) ? data[idx] : 0;
        }
        uint32_t blk[16];
        prl_b3_parse_block(buf, blk);
        uint32_t o16[16];
        prl_b3_compress_full(in, blk, o16);
        for (int i = 0; i < 8; ++i) {
            out32[4 * i + 0] = (uint8_t)(o16[i]);
            out32[4 * i + 1] = (uint8_t)(o16[i] >> 8);
            out32[4 * i + 2] = (uint8_t)(o16[i] >> 16);
            out32[4 * i + 3] = (uint8_t)(o16[i] >> 24);
        }
    }
}

// ---- BLAKE3 tree (для входов > 1024 байт), spec reference ----
//
// Точная трансляция Reference::blake3_tree / output_parent_cv из
// oconnor67/blake3 reference.c. Разбиение: левое поддерево = наибольшая
// степень двойки, СТРОГО меньшая общей длины; counter родителя =
// start_chunk + left_chunks (чанки считаются от 0 по всему входу).
// Для pearlhash нужен только single-chunk путь (<=1024B) выше; этот код
// — общий фолбэк и документация формата.

__device__ __forceinline__ uint64_t prl_b3_left_subtree_len(uint64_t len) {
    uint64_t q = 1ull;
    while (q * 2ull < len) q *= 2ull;  // наибольшая 2^k < len
    return q;
}

__device__ __forceinline__ void prl_b3_parent_node(const uint32_t left[8],
                                                   const uint32_t right[8],
                                                   uint64_t counter,
                                                   const uint32_t key[8],
                                                   uint32_t kind_flags,
                                                   bool is_root,
                                                   uint32_t cv_out[8]) {
    uint32_t blk[16];
    for (int i = 0; i < 8; ++i) blk[i] = left[i];
    for (int i = 0; i < 8; ++i) blk[8 + i] = right[i];
    uint32_t flags = PRL_B3_PARENT_START | PRL_B3_PARENT_END | kind_flags;
    if (is_root) flags |= PRL_B3_ROOT;
    PrlB3Input in = prl_b3_new_input(key, counter, 64, flags);
    uint32_t o16[16];
    prl_b3_compress_full(in, blk, o16);
    for (int i = 0; i < 8; ++i) cv_out[i] = o16[i];  // low half == CV/root
}

// Итеративный root-хеш произвольной длины (в т.ч. дерева чанков), один поток.
__device__ __forceinline__ void prl_b3_hash_any(const uint8_t *data,
                                                uint64_t len,
                                                const uint32_t key[8],
                                                uint32_t kind_flags,
                                                uint8_t out32[32]) {
    if (len <= 1024) {
        prl_b3_hash_small(data, (uint32_t)len, key, kind_flags, out32);
        return;
    }
    struct Frame {
        uint64_t start;  // стартовый байт поддерева
        uint64_t len;    // длина поддерева
        int stage;       // 0=идём влево, 1=идём вправо, 2=собираем
        uint32_t left[8];
    };
    Frame stack[64];
    int sp = 0;
    stack[0].start = 0;
    stack[0].len = len;
    stack[0].stage = 0;

    uint32_t done[8];  // CV/leaf-hash только что завершённого узла
    bool have_done = false;

    for (;;) {
        Frame &f = stack[sp];
        if (have_done) {
            if (f.stage == 0) {
                // завершилось ЛЕВОЕ поддерево — сохраняем и идём в правое
                for (int i = 0; i < 8; ++i) f.left[i] = done[i];
                f.stage = 1;
                have_done = false;
            } else {
                // f.stage == 1: завершилось ПРАВОЕ — схлопываем кадр
                const uint64_t q = prl_b3_left_subtree_len(f.len);
                const uint64_t parent_counter = f.start / 1024ull + q / 1024ull;
                const bool is_root = (sp == 0);
                uint32_t cvout[8];
                prl_b3_parent_node(f.left, done, parent_counter, key,
                                   kind_flags, is_root, cvout);
                for (int i = 0; i < 8; ++i) done[i] = cvout[i];
                sp--;
                if (sp < 0) {
                    for (int i = 0; i < 8; ++i) {
                        out32[4 * i + 0] = (uint8_t)done[i];
                        out32[4 * i + 1] = (uint8_t)(done[i] >> 8);
                        out32[4 * i + 2] = (uint8_t)(done[i] >> 16);
                        out32[4 * i + 3] = (uint8_t)(done[i] >> 24);
                    }
                    return;
                }
                // возврат из правого ребёнка родителя -> родитель собирается
                continue;
            }
        }
        if (f.stage == 0) {
            const uint64_t q = prl_b3_left_subtree_len(f.len);
            if (q <= 1024ull) {
                uint8_t h[32];
                prl_b3_hash_small(data + f.start, (uint32_t)q, key, kind_flags,
                                  h);
                for (int i = 0; i < 8; ++i) done[i] = prl_b3_load32(h + 4 * i);
                have_done = true;
                continue;
            }
            ++sp;
            stack[sp].start = f.start;
            stack[sp].len = q;
            stack[sp].stage = 0;
            continue;
        }
        // stage == 1: левый ребёнок уже во f.left, идём в правый
        const uint64_t q = prl_b3_left_subtree_len(f.len);
        const uint64_t rle = f.len - q;
        if (rle <= 1024ull) {
            uint8_t h[32];
            prl_b3_hash_small(data + f.start + q, (uint32_t)rle, key,
                              kind_flags, h);
            for (int i = 0; i < 8; ++i) done[i] = prl_b3_load32(h + 4 * i);
            have_done = true;
            continue;
        }
        ++sp;
        stack[sp].start = f.start + q;
        stack[sp].len = rle;
        stack[sp].stage = 0;
    }
}

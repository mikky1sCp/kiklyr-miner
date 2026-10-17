// pearlhash fused miner kernel: один поток = одна попытка (grid/nonce).
//
// Путь полностью на устройстве, bit-exact с NumPy-эталоном core.py:
//   seed  = BLAKE3(key || nonce_be64)[:8] LE            (derive_seed)
//   A,B   ~ U[elem_min, elem_max] Philox salt 0/1       (row-major flat)
//   V,Y   ~ U[-8,8] salt 3/4                            ([K,R], col j row-major)
//   E_ij  = clip((U_i . V_col_j >> 6) << 6)             (wrap_lowrank_noise)
//   Ap    = wrap8(A + E), Bp = wrap8(B + F)             (int8 mod 256)
//   C'    = Ap @ Bp^T                                   (INT32 accumulate точный)
//   fold  = XOR_tiles BLAKE3(tile_le_i32_bytes)         (transcript_fold)
//   H     = BLAKE3(fold);  решение если int_le(H) < target
//
// Порядок потребления слов Philox в точности повторяет randint_range
// эталона: каждое значение — следующее слово потока роли; rejection
// расходует слова последовательно.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#include "philox_cuda.cuh"
#include "blake3_cuda.cuh"

#define PRL_NOISE_LO (-8)
#define PRL_NOISE_HI 8
#define PRL_QUANT_SHIFT 6

namespace {

__device__ __forceinline__ uint64_t prl_dev_seed(const uint8_t *key,
                                                 uint32_t klen,
                                                 uint64_t nonce) {
    // key <= 96B; nonce big-endian appended -> <= 104B <= 1024 => single chunk
    uint8_t buf[104];
    for (uint32_t i = 0; i < klen; ++i) buf[i] = key[i];
    for (int i = 0; i < 8; ++i)
        buf[klen + i] = (uint8_t)(nonce >> (56 - 8 * i));
    uint32_t ek[8];
    prl_b3_empty_key(ek);
    uint8_t h[32];
    prl_b3_hash_small(buf, klen + 8, ek, 0, h);
    uint64_t s = 0;
    for (int i = 0; i < 8; ++i) s |= (uint64_t)h[i] << (8 * i);
    return s;
}

template <int M, int N, int K, int R, int TH, int TW, int LO, int HI>
__global__ void prl_fused_kernel(const uint8_t *__restrict__ key,
                                 const uint32_t klen,
                                 const uint64_t *__restrict__ nonces,
                                 const int count,
                                 const uint64_t *__restrict__ target_words,
                                 int64_t *__restrict__ out_nonce,
                                 uint8_t *__restrict__ out_hash,
                                 int *__restrict__ found_ctr,
                                 const int max_solutions) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    const uint64_t nonce = nonces[tid];
    const uint64_t seed = prl_dev_seed(key, klen, nonce);

    uint32_t kA0, kA1, kB0, kB1, kU0, kU1, kV0, kV1, kX0, kX1, kY0, kY1;
    prl_role_key(seed, 0, kA0, kA1);
    prl_role_key(seed, 1, kB0, kB1);
    prl_role_key(seed, 2, kU0, kU1);
    prl_role_key(seed, 3, kV0, kV1);
    prl_role_key(seed, 4, kX0, kX1);
    prl_role_key(seed, 5, kY0, kY1);
    uint64_t cwA = 0, cwB = 0, cwU = 0, cwV = 0, cwX = 0, cwY = 0;

    // --- шумовые операнды: столбцы V и Y ([K,R] row-major -> V[k][r]) ---
    int8_t Vr[K][R], Yr[K][R];
    for (int kk = 0; kk < K; ++kk)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            Vr[kk][r] = (int8_t)prl_reject_next(kV0, kV1, &cwV, PRL_NOISE_LO, PRL_NOISE_HI);
            Yr[kk][r] = (int8_t)prl_reject_next(kY0, kY1, &cwY, PRL_NOISE_LO, PRL_NOISE_HI);
        }

    // --- Ap = wrap8(A + clip_q(U . Vcol)) ; построчно, word-order эталона ---
    int8_t Ap[M][K], Bp[N][K];
    for (int i = 0; i < M; ++i) {
#pragma unroll
        for (int j = 0; j < K; ++j) {
            const int a = prl_reject_next(kA0, kA1, &cwA, LO, HI);
            int64_t q = 0;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int u = prl_reject_next(kU0, kU1, &cwU, PRL_NOISE_LO, PRL_NOISE_HI);
                q += (int64_t)u * (int64_t)Vr[j][r];
            }
            int64_t e = ((q >> PRL_QUANT_SHIFT) << PRL_QUANT_SHIFT);
            if (e > 127) e = 127;
            if (e < -128) e = -128;
            Ap[i][j] = (int8_t)((a + e + 128) & 255) - 128;
        }
    }
    // --- Bp = wrap8(B + clip_q(X . Ycol)) ---
    for (int i = 0; i < N; ++i) {
#pragma unroll
        for (int j = 0; j < K; ++j) {
            const int b = prl_reject_next(kB0, kB1, &cwB, LO, HI);
            int64_t q = 0;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int x = prl_reject_next(kX0, kX1, &cwX, PRL_NOISE_LO, PRL_NOISE_HI);
                q += (int64_t)x * (int64_t)Yr[j][r];
            }
            int64_t f = ((q >> PRL_QUANT_SHIFT) << PRL_QUANT_SHIFT);
            if (f > 127) f = 127;
            if (f < -128) f = -128;
            Bp[i][j] = (int8_t)((b + f + 128) & 255) - 128;
        }
    }

    // --- transcript fold поверх C' тайлами TH x TW (row-major tiles) ---
    constexpr int TILES_H = M / TH;
    constexpr int TILES_W = N / TW;
    uint32_t ek[8];
    prl_b3_empty_key(ek);
    uint8_t fold[32];
    for (int x = 0; x < 32; ++x) fold[x] = 0;

    for (int ti = 0; ti < TILES_H; ++ti) {
        for (int tj = 0; tj < TILES_W; ++tj) {
            uint8_t tbuf[TH * TW * 4];
            for (int a = 0; a < TH; ++a) {
                const int gi = ti * TH + a;
                for (int b = 0; b < TW; ++b) {
                    const int gj = tj * TW + b;
                    int32_t c = 0;
#pragma unroll 8
                    for (int j = 0; j < K; ++j)
                        c += (int32_t)Ap[gi][j] * (int32_t)Bp[gj][j];
                    const int off = (a * TW + b) * 4;
                    tbuf[off + 0] = (uint8_t)(c);
                    tbuf[off + 1] = (uint8_t)(c >> 8);
                    tbuf[off + 2] = (uint8_t)(c >> 16);
                    tbuf[off + 3] = (uint8_t)(c >> 24);
                }
            }
            uint8_t hh[32];
            prl_b3_hash_small(tbuf, TH * TW * 4, ek, 0, hh);
            for (int x = 0; x < 32; ++x) fold[x] ^= hh[x];
        }
    }

    uint8_t H[32];
    prl_b3_hash_small(fold, 32, ek, 0, H);

    // сравнение int_le(H) < target (target — 4 слова uint64 little-endian)
    uint64_t hv[4];
    for (int w = 0; w < 4; ++w) {
        hv[w] = 0;
        for (int i = 0; i < 8; ++i)
            hv[w] |= (uint64_t)H[8 * w + i] << (8 * i);
    }
    bool lt = false;
    for (int w = 3; w >= 0 && !lt; --w)
        if (hv[w] < target_words[w]) lt = true;
    if (lt) {
        const int slot = atomicAdd(found_ctr, 1);
        if (slot < max_solutions) {
            out_nonce[slot] = (int64_t)nonce;
            for (int i = 0; i < 32; ++i) out_hash[slot * 32 + i] = H[i];
        }
    }
}

// Ядро-сверка для unit-тестов: hash_grid без поиска (возвращает H).
template <int M, int N, int K, int R, int TH, int TW, int LO, int HI>
__global__ void prl_hash_kernel(const uint8_t *__restrict__ key,
                                const uint32_t klen,
                                const uint64_t *__restrict__ nonces,
                                const int count,
                                uint8_t *__restrict__ out_hashes) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    const uint64_t nonce = nonces[tid];
    const uint64_t seed = prl_dev_seed(key, klen, nonce);

    uint32_t kA0, kA1, kB0, kB1, kU0, kU1, kV0, kV1, kX0, kX1, kY0, kY1;
    prl_role_key(seed, 0, kA0, kA1);
    prl_role_key(seed, 1, kB0, kB1);
    prl_role_key(seed, 2, kU0, kU1);
    prl_role_key(seed, 3, kV0, kV1);
    prl_role_key(seed, 4, kX0, kX1);
    prl_role_key(seed, 5, kY0, kY1);
    uint64_t cwA = 0, cwB = 0, cwU = 0, cwV = 0, cwX = 0, cwY = 0;

    int8_t Vr[K][R], Yr[K][R];
    for (int kk = 0; kk < K; ++kk)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            Vr[kk][r] = (int8_t)prl_reject_next(kV0, kV1, &cwV, PRL_NOISE_LO, PRL_NOISE_HI);
            Yr[kk][r] = (int8_t)prl_reject_next(kY0, kY1, &cwY, PRL_NOISE_LO, PRL_NOISE_HI);
        }
    int8_t Ap[M][K], Bp[N][K];
    for (int i = 0; i < M; ++i)
#pragma unroll
        for (int j = 0; j < K; ++j) {
            const int a = prl_reject_next(kA0, kA1, &cwA, LO, HI);
            int64_t q = 0;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int u = prl_reject_next(kU0, kU1, &cwU, PRL_NOISE_LO, PRL_NOISE_HI);
                q += (int64_t)u * (int64_t)Vr[j][r];
            }
            int64_t e = ((q >> PRL_QUANT_SHIFT) << PRL_QUANT_SHIFT);
            if (e > 127) e = 127;
            if (e < -128) e = -128;
            Ap[i][j] = (int8_t)((a + e + 128) & 255) - 128;
        }
    for (int i = 0; i < N; ++i)
#pragma unroll
        for (int j = 0; j < K; ++j) {
            const int b = prl_reject_next(kB0, kB1, &cwB, LO, HI);
            int64_t q = 0;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int x = prl_reject_next(kX0, kX1, &cwX, PRL_NOISE_LO, PRL_NOISE_HI);
                q += (int64_t)x * (int64_t)Yr[j][r];
            }
            int64_t f = ((q >> PRL_QUANT_SHIFT) << PRL_QUANT_SHIFT);
            if (f > 127) f = 127;
            if (f < -128) f = -128;
            Bp[i][j] = (int8_t)((b + f + 128) & 255) - 128;
        }

    constexpr int TILES_H = M / TH;
    constexpr int TILES_W = N / TW;
    uint32_t ek[8];
    prl_b3_empty_key(ek);
    uint8_t fold[32];
    for (int x = 0; x < 32; ++x) fold[x] = 0;
    for (int ti = 0; ti < TILES_H; ++ti)
        for (int tj = 0; tj < TILES_W; ++tj) {
            uint8_t tbuf[TH * TW * 4];
            for (int a = 0; a < TH; ++a) {
                const int gi = ti * TH + a;
                for (int b = 0; b < TW; ++b) {
                    const int gj = tj * TW + b;
                    int32_t c = 0;
#pragma unroll 8
                    for (int j = 0; j < K; ++j)
                        c += (int32_t)Ap[gi][j] * (int32_t)Bp[gj][j];
                    const int off = (a * TW + b) * 4;
                    tbuf[off + 0] = (uint8_t)(c);
                    tbuf[off + 1] = (uint8_t)(c >> 8);
                    tbuf[off + 2] = (uint8_t)(c >> 16);
                    tbuf[off + 3] = (uint8_t)(c >> 24);
                }
            }
            uint8_t hh[32];
            prl_b3_hash_small(tbuf, TH * TW * 4, ek, 0, hh);
            for (int x = 0; x < 32; ++x) fold[x] ^= hh[x];
        }
    prl_b3_hash_small(fold, 32, ek, 0, out_hashes + 32 * tid);
}

struct ShapeKey {
    int m, n, k, r, th, tw, lo, hi;
};

template <int M, int N, int K, int R, int TH, int TW, int LO, int HI>
void launch_hash(const ShapeKey &, torch::Tensor &key, torch::Tensor &nonces,
                 torch::Tensor &out, int count, cudaStream_t stream) {
    const int threads = 128;
    const int blocks = (count + threads - 1) / threads;
    prl_hash_kernel<M, N, K, R, TH, TW, LO, HI><<<blocks, threads, 0, stream>>>(
        key.data_ptr<uint8_t>(), (uint32_t)key.numel(),
        nonces.data_ptr<int64_t>(), count, out.data_ptr<uint8_t>());
}

template <int M, int N, int K, int R, int TH, int TW, int LO, int HI>
void launch_search(const ShapeKey &, torch::Tensor &key, torch::Tensor &nonces,
                   torch::Tensor &target, torch::Tensor &out_nonce,
                   torch::Tensor &out_hash, torch::Tensor &found, int count,
                   int max_solutions, cudaStream_t stream) {
    const int threads = 128;
    const int blocks = (count + threads - 1) / threads;
    prl_fused_kernel<M, N, K, R, TH, TW, LO, HI><<<blocks, threads, 0, stream>>>(
        key.data_ptr<uint8_t>(), (uint32_t)key.numel(),
        nonces.data_ptr<int64_t>(), count, target.data_ptr<uint64_t>(),
        out_nonce.data_ptr<int64_t>(), out_hash.data_ptr<uint8_t>(),
        found.data_ptr<int>(), max_solutions);
}

// Таблица допустимых форм (размеры кратны тайлам; компактные конфигурации
// для тестов + референсная 64/64/64/R16/tile16).
#define PRL_FOREACH_SHAPE(F)                                                    \
    F(64, 64, 64, 16, 16, 16, -64, 64)                                          \
    F(64, 64, 64, 16, 64, 64, -64, 64)                                          \
    F(32, 32, 32, 8, 8, 8, -64, 64)                                             \
    F(16, 16, 16, 4, 8, 8, -64, 64)                                             \
    F(16, 16, 16, 4, 16, 16, -64, 64)

bool shape_supported(int64_t m, int64_t n, int64_t k, int64_t r, int64_t th,
                     int64_t tw, int64_t lo, int64_t hi) {
    ShapeKey sk{(int)m, (int)n, (int)k, (int)r, (int)th, (int)tw, (int)lo,
                (int)hi};
#define TRY(M_, N_, K_, R_, TH_, TW_, LO_, HI_)                                 \
    if (sk.m == M_ && sk.n == N_ && sk.k == K_ && sk.r == R_ && sk.th == TH_ && \
        sk.tw == TW_ && sk.lo == LO_ && sk.hi == HI_)                           \
        return true;
    PRL_FOREACH_SHAPE(TRY)
#undef TRY
    return false;
}

bool dispatch_hash(const ShapeKey &sk, torch::Tensor &key,
                   torch::Tensor &nonces, torch::Tensor &out, int count,
                   cudaStream_t stream) {
    auto st = sk;
#define TRY(M_, N_, K_, R_, TH_, TW_, LO_, HI_)                                 \
    if (st.m == M_ && st.n == N_ && st.k == K_ && st.r == R_ && st.th == TH_ && \
        st.tw == TW_ && st.lo == LO_ && st.hi == HI_) {                          \
        launch_hash<M_, N_, K_, R_, TH_, TW_, LO_, HI_>(st, key, nonces, out,    \
                                                        count, stream);          \
        return true;                                                             \
    }
    PRL_FOREACH_SHAPE(TRY)
#undef TRY
    return false;
}

bool dispatch_search(const ShapeKey &sk, torch::Tensor &key,
                     torch::Tensor &nonces, torch::Tensor &target,
                     torch::Tensor &out_nonce, torch::Tensor &out_hash,
                     torch::Tensor &found, int count, int max_solutions,
                     cudaStream_t stream) {
    auto st = sk;
#define TRY(M_, N_, K_, R_, TH_, TW_, LO_, HI_)                                 \
    if (st.m == M_ && st.n == N_ && st.k == K_ && st.r == R_ && st.th == TH_ && \
        st.tw == TW_ && st.lo == LO_ && st.hi == HI_) {                          \
        launch_search<M_, N_, K_, R_, TH_, TW_, LO_, HI_>(                       \
            st, key, nonces, target, out_nonce, out_hash, found, count,          \
            max_solutions, stream);                                              \
        return true;                                                             \
    }
    PRL_FOREACH_SHAPE(TRY)
#undef TRY
    return false;
}

std::vector<torch::Tensor> hash_batch(torch::Tensor key, torch::Tensor nonces,
                                      int64_t m, int64_t n, int64_t k,
                                      int64_t r, int64_t th, int64_t tw,
                                      int64_t lo, int64_t hi) {
    TORCH_CHECK(nonces.is_cuda() && nonces.scalar_type() == torch::kLong);
    TORCH_CHECK(key.is_cuda() && key.scalar_type() == torch::kByte);
    TORCH_CHECK(key.numel() <= 96, "job key должен быть <= 96 байт");
    const int count = (int)nonces.numel();
    auto out = torch::zeros({count, 32},
                            torch::TensorOptions().dtype(torch::kByte)
                                .device(nonces.device()));
    ShapeKey sk{(int)m, (int)n, (int)k, (int)r, (int)th, (int)tw, (int)lo,
                (int)hi};
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(dispatch_hash(sk, key, nonces, out, count, stream),
                "конфигурация не скомпилирована: ", m, ",", n, ",", k, ",",
                r, ",", th, ",", tw, ",", lo, ",", hi);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out};
}

std::vector<torch::Tensor> search_batch(
    torch::Tensor key, torch::Tensor nonces, torch::Tensor target_words,
    int64_t m, int64_t n, int64_t k, int64_t r, int64_t th, int64_t tw,
    int64_t lo, int64_t hi, int64_t max_solutions) {
    TORCH_CHECK(nonces.is_cuda() && nonces.scalar_type() == torch::kLong);
    TORCH_CHECK(key.is_cuda() && key.scalar_type() == torch::kByte);
    TORCH_CHECK(target_words.is_cuda() &&
                target_words.scalar_type() == torch::kLong &&
                target_words.numel() == 4);
    const int count = (int)nonces.numel();
    auto dev = nonces.device();
    auto out_nonce = torch::zeros(
        {max_solutions}, torch::TensorOptions().dtype(torch::kLong).device(dev));
    auto out_hash = torch::zeros(
        {max_solutions, 32}, torch::TensorOptions().dtype(torch::kByte).device(dev));
    auto found =
        torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt).device(dev));
    ShapeKey sk{(int)m, (int)n, (int)k, (int)r, (int)th, (int)tw, (int)lo,
                (int)hi};
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(dispatch_search(sk, key, nonces, target_words, out_nonce,
                                out_hash, found, count, (int)max_solutions,
                                stream),
                "конфигурация не скомпилирована");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out_nonce, out_hash, found};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("shape_supported", &shape_supported,
          "is this (m,n,k,r,tile,range) configuration compiled into the ext");
    m.def("hash_batch", &hash_batch, "pearlhash hash_grid batch (fused CUDA)");
    m.def("search_batch", &search_batch,
          "pearlhash nonce search batch (fused CUDA)");
}

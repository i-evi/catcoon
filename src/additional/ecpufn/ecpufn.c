#include <string.h>

#ifdef __x86_64
#include <immintrin.h>
#endif

#include "ecpufn.h"

#define NAIVE_CONV2D_IMPLEMENTATION(dtype) \
void naive_conv2d_ ## dtype (const dtype *in,     \
    dtype *out, i32 ix, i32 iy, i32 ox, i32 oy,   \
  i32 sx, i32 sy, const dtype *k, i32 kw)         \
{                                                 \
  i32 i, j, u, v;                                 \
  dtype acc;                                      \
  for (i = 0; i <= iy - kw; i += sy) {            \
    for (j = 0; j <= ix - kw; j += sx) {          \
      acc = 0;                                    \
      for (u = 0; u < kw; ++u) {                  \
        for (v = 0; v < kw; ++v) {                \
          acc += *(in + (i + u) * ix + (j + v)) * \
                 *(k + u * kw + v);               \
        }                                         \
      }                                           \
      *out++ = acc;                               \
    }                                             \
  }                                               \
}

NAIVE_CONV2D_IMPLEMENTATION  (i8);
NAIVE_CONV2D_IMPLEMENTATION  (u8);
NAIVE_CONV2D_IMPLEMENTATION  (i16);
NAIVE_CONV2D_IMPLEMENTATION  (u16);
NAIVE_CONV2D_IMPLEMENTATION  (i32);
NAIVE_CONV2D_IMPLEMENTATION  (u32);
NAIVE_CONV2D_IMPLEMENTATION  (i64);
NAIVE_CONV2D_IMPLEMENTATION  (u64);
NAIVE_CONV2D_IMPLEMENTATION  (f32);
NAIVE_CONV2D_IMPLEMENTATION  (f64);

#define DFL_CONV2D_IMPLEMENTATION(dtype) \
void ecpu_conv2d_ ## dtype (const dtype *in, dtype *out, i32 ix, i32 iy, \
	i32 ox, i32 oy, i32 sx, i32 sy, const dtype *k, i32 kw)          \
{                                                                        \
	naive_conv2d_ ## dtype(in, out, ix, iy, ox, oy, sx, sy, k, kw);  \
}

DFL_CONV2D_IMPLEMENTATION  (i8);
DFL_CONV2D_IMPLEMENTATION  (u8);
DFL_CONV2D_IMPLEMENTATION  (i16);
DFL_CONV2D_IMPLEMENTATION  (u16);
DFL_CONV2D_IMPLEMENTATION  (i32);
DFL_CONV2D_IMPLEMENTATION  (u32);
DFL_CONV2D_IMPLEMENTATION  (i64);
DFL_CONV2D_IMPLEMENTATION  (u64);
/* DFL_CONV2D_IMPLEMENTATION  (f32); */
DFL_CONV2D_IMPLEMENTATION  (f64);

void ecpu_conv2d_f32(const f32 *in, f32 *out, i32 ix, i32 iy,
	i32 ox, i32 oy, i32 sx, i32 sy, const f32 *k, i32 kw)
{
	switch (kw) {
#ifdef ALT_CONV2D_F32_K1S1
	case 1:
		if (sx == sy && sx == 1)
			ALT_CONV2D_F32_K1S1(
				in, out, ix, iy, ox, oy, sx, sy, k, kw);
		else
			naive_conv2d_f32(
				in, out, ix, iy, ox, oy, sx, sy, k, kw);
		break;
#endif
#ifdef ALT_CONV2D_F32_K2SX
	case 2:
		ALT_CONV2D_F32_K2SX(in, out, ix, iy, ox, oy, sx, sy, k, kw);
		break;
#endif
#ifdef ALT_CONV2D_F32_K3SX
	case 3:
		ALT_CONV2D_F32_K3SX(in, out, ix, iy, ox, oy, sx, sy, k, kw);
		break;
#endif
	default:
		naive_conv2d_f32(in, out, ix, iy, ox, oy, sx, sy, k, kw);
		break;
	}
}

#if defined(__x86_64) && defined(__SSE4_1__)
void sse4_conv2d_f32_k1s1(const f32 *in, f32 *out, i32 ix, i32 iy,
	i32 ox, i32 oy, i32 sx, i32 sy, const f32 *k, i32 kw)
{
	i32 i, size, nblk;
	__m128 lvk, blk;
	lvk = _mm_set1_ps(*k);
	size = ix * iy;
	nblk = size >> 2;
	for (i = 0; i < nblk; ++i) {
		blk = _mm_loadu_ps(in);
		blk = blk * lvk;
		_mm_storeu_ps(out, blk);
		in  += 4;
		out += 4;
	}
	size = size - (nblk << 2);
	for (i = 0; i < size; ++i) {
		*out++ = *in++ * *k;
	}
}

void sse4_conv2d_f32_k2sx(const f32 *in, f32 *out, i32 ix, i32 iy,
	i32 ox, i32 oy, i32 sx, i32 sy, const f32 *k, i32 kw)
{
	i32 i, j, off;
	__m128 lvk, blk;
	__m128 pdv = _mm_set_ps(1., 1., 1., 1.);
	f32 *res = (f32*)&(blk);
	lvk = _mm_set_ps(k[3], k[2], k[1], k[0]);
	for (i = 0; i <= iy - kw; i += sy) {
		for (j = 0; j <= ix - kw; j += sx) {
			off = i * ix + j;
			blk = _mm_shuffle_ps(
				_mm_loadu_ps(in + off),          // l2
				_mm_loadu_ps(in + off + ix - 2), // h2
				_MM_SHUFFLE(3, 2, 1, 0));
			blk = blk * lvk;
			blk = _mm_dp_ps(blk, pdv, 0xF1);
			*out++ = *res;
		}
	}
}

void sse4_conv2d_f32_k3sx(const f32 *in, f32 *out, i32 ix, i32 iy,
	i32 ox, i32 oy, i32 sx, i32 sy, const f32 *k, i32 kw)
{
	i32 i, j, u, off;
	__m128 lvk[3];
	__m128 blk, acc;
	__m128 pdvr = _mm_set_ps(0., 1., 1., 1.);
	__m128 pdvc = _mm_set_ps(1., 1., 1., 0.);
	f32 *res = (f32*)&(acc);
	for (u = 0; u < kw; ++u) {
		memcpy(&(lvk[u]), k + u * kw, sizeof(f32) * kw);
	}
	for (i = 0; i <= iy - kw - sy; i += sy) {
		for (j = 0; j <= ix - kw; j += sx) {
			acc = _mm_setzero_ps();
			off = i * ix + j;
			for (u = 0; u < kw; ++u) {
				blk = _mm_loadu_ps(in + off);
				acc += blk * lvk[u];
				off += ix;
			}
			acc = _mm_dp_ps(acc, pdvr, 0xF1);
			*out++ = *res;
		}
	}
	if (iy - kw >= i) {
		off = i * ix;
		for (u = 0; u < kw; ++u) {
			memcpy(((f32*)&(lvk[u])) + 1,
				k + u * kw, sizeof(f32) * kw);
		}
		for (j = -1; j <= ix - kw - 1; j += sx) {
			acc = _mm_setzero_ps();
			off = i * ix + j;
			for (u = 0; u < kw; ++u) {
				blk = _mm_loadu_ps(in + off);
				acc += blk * lvk[u];
				off += ix;
			}
			acc = _mm_dp_ps(acc, pdvc, 0xF1);
			*out++ = *res;
		}
	}
}
#endif /* __SSE4_1__ */

#define NAIVE_DOTPROD_IMPLEMENTATION(dtype) \
void naive_dot_prod_ ## dtype (const dtype *in, \
	dtype *out, const dtype *w, i32 iw)     \
{                                               \
	i32 i;                                  \
	*out = 0;                               \
	for (i = 0; i < iw; ++i) {              \
		*out += *(in + i) * (*(w + i)); \
	}                                       \
}

NAIVE_DOTPROD_IMPLEMENTATION  (i8);
NAIVE_DOTPROD_IMPLEMENTATION  (u8);
NAIVE_DOTPROD_IMPLEMENTATION  (i16);
NAIVE_DOTPROD_IMPLEMENTATION  (u16);
NAIVE_DOTPROD_IMPLEMENTATION  (i32);
NAIVE_DOTPROD_IMPLEMENTATION  (u32);
NAIVE_DOTPROD_IMPLEMENTATION  (i64);
NAIVE_DOTPROD_IMPLEMENTATION  (u64);
NAIVE_DOTPROD_IMPLEMENTATION  (f32);
NAIVE_DOTPROD_IMPLEMENTATION  (f64);

#define DFL_DOTPROD_IMPLEMENTATION(dtype) \
void ecpu_dot_prod_ ## dtype (const dtype *in,     \
	dtype *out, const dtype *w, i32 iw)        \
{                                                  \
	naive_dot_prod_ ## dtype (in, out, w, iw); \
}

DFL_DOTPROD_IMPLEMENTATION  (i8);
DFL_DOTPROD_IMPLEMENTATION  (u8);
DFL_DOTPROD_IMPLEMENTATION  (i16);
DFL_DOTPROD_IMPLEMENTATION  (u16);
DFL_DOTPROD_IMPLEMENTATION  (i32);
DFL_DOTPROD_IMPLEMENTATION  (u32);
DFL_DOTPROD_IMPLEMENTATION  (i64);
DFL_DOTPROD_IMPLEMENTATION  (u64);
/* DFL_DOTPROD_IMPLEMENTATION  (f32); */
DFL_DOTPROD_IMPLEMENTATION  (f64);

void ecpu_dot_prod_f32(const f32 *in, f32 *out, const f32 *w, i32 iw)
{
#ifdef ALT_DOTPROD_F32
	ALT_DOTPROD_F32(in, out, w, iw);
#else
	naive_dot_prod_f32(in, out, w, iw);
#endif
}

#if defined(__x86_64) && defined(__SSE4_1__)
void sse4_dot_prod_f32(const f32 *in, f32 *out, const f32 *w, i32 iw)
{
	i32 i;
	__m128 blk, acc, vwb;
	__m128 pdv = _mm_set1_ps(1.);
	f32 *res = (f32*)&(acc);
	acc = _mm_setzero_ps();
	for (i = 0; i < iw - 4; i += 4) {
		blk = _mm_loadu_ps(in + i);
		vwb = _mm_loadu_ps( w + i);
		acc += blk * vwb;
	}
	acc = _mm_dp_ps(acc, pdv, 0xF1);
	*out = *res;
	for (; i < iw; ++i) {
		*out += *(in + i) * (*(w + i));
	}
}
#endif

#if defined(__x86_64) && defined(__AVX__)
void avx_dot_prod_f32(const f32 *in, f32 *out, const f32 *w, i32 iw)
{
	i32 i;
	__m256 blk, acc, vwb;
	__m256 pdv = _mm256_set1_ps(1.);
	f32 *resl = ((f32*)&(acc)) + 0;
	f32 *resh = ((f32*)&(acc)) + 4;
	acc = _mm256_setzero_ps();
	for (i = 0; i < iw - 8; i += 8) {
		blk = _mm256_loadu_ps(in + i);
		vwb = _mm256_loadu_ps( w + i);
		acc += blk * vwb;
	}
	acc = _mm256_dp_ps(acc, pdv, 0xF1);
	*out = *resl + *resh;
	for (; i < iw; ++i) {
		*out += *(in + i) * (*(w + i));
	}
}
#endif

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
		if (sx == sy && sx == 1)
			ALT_CONV2D_F32_K3S1(
				in, out, ix, iy, ox, oy, sx, sy, k, kw);
		else
			ALT_CONV2D_F32_K3SX(
				in, out, ix, iy, ox, oy, sx, sy, k, kw);
		break;
#endif
	default:
		naive_conv2d_f32(in, out, ix, iy, ox, oy, sx, sy, k, kw);
		break;
	}
}

#if defined(__x86_64) && defined(__SSE__)
void sse_conv2d_f32_k1s1(const f32 *in, f32 *out, i32 ix, i32 iy,
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

void sse_conv2d_f32_k2sx(const f32 *in, f32 *out, i32 ix, i32 iy,
	i32 ox, i32 oy, i32 sx, i32 sy, const f32 *k, i32 kw)
{
	i32 i, j, off;
	__m128 lvk, blk;
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
			*res += res[1] + res[2] + res[3];
			*out++ = *res;
		}
	}
}

void sse_conv2d_f32_k3s1(const f32 *in, f32 *out, i32 ix, i32 iy,
	i32 ox, i32 oy, i32 sx, i32 sy, const f32 *k, i32 kw)
{
	i32 i, j, u, off;
	__m128 vkl[3];
	__m128 vkh[3];
	__m128 blk, accl, acch;
	f32 *resl = (f32*)&(accl);
	f32 *resh = (f32*)&(acch);
	for (u = 0; u < kw; ++u) {
		memcpy(((f32*)&(vkl[u])) + 0, k + u * kw, sizeof(f32) * kw);
		memcpy(((f32*)&(vkh[u])) + 1, k + u * kw, sizeof(f32) * kw);
	}
	for (i = 0; i <= iy - kw - sy; i += sy) {
		for (j = 0; j < ix - kw; j += 2) {
			accl = _mm_setzero_ps();
			acch = _mm_setzero_ps();
			off = i * ix + j;
			for (u = 0; u < kw; ++u) {
				blk = _mm_loadu_ps(in + off);
				accl += blk * vkl[u];
				acch += blk * vkh[u];
				off += ix;
			}
			*out++ = (resl[0] + resl[1] + resl[2]);
			*out++ = (resh[1] + resh[2] + resh[3]);
		}
		if (j == ix - kw) {
			accl = _mm_setzero_ps();
			off = i * ix + j;
			for (u = 0; u < kw; ++u) {
				blk = _mm_loadu_ps(in + off);
				accl += blk * vkl[u];
				off  += ix;
			}
			*out++ = (resl[0] + resl[1] + resl[2]);
		}
	}
	if (iy - kw >= i) {
		off = i * ix;
		for (j = -1; j <= ix - kw - 1; ++j) {
			acch = _mm_setzero_ps();
			off = i * ix + j;
			for (u = 0; u < kw; ++u) {
				blk = _mm_loadu_ps(in + off);
				acch += blk * vkh[u];
				off  += ix;
			}
			*out++ = (resh[1] + resh[2] + resh[3]);
		}
	}
}

void sse_conv2d_f32_k3sx(const f32 *in, f32 *out, i32 ix, i32 iy,
	i32 ox, i32 oy, i32 sx, i32 sy, const f32 *k, i32 kw)
{
	i32 i, j, u, off;
	__m128 lvk[3];
	__m128 blk, acc;
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
			*out++ = (res[0] + res[1] + res[2]);
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
			*out++ = (res[1] + res[2] + res[3]);
		}
	}
}
#endif /* __SSE__ */

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

#if defined(__x86_64) && defined(__SSE__)
void sse_dot_prod_f32(const f32 *in, f32 *out, const f32 *w, i32 iw)
{
	i32 i;
	__m128 blk, acc, vwb;
	f32 *res = (f32*)&(acc);
	acc = _mm_setzero_ps();
	for (i = 0; i < iw - 4; i += 4) {
		blk = _mm_loadu_ps(in + i);
		vwb = _mm_loadu_ps( w + i);
		acc += blk * vwb;
	}
	*res += res[1] + res[2] + res[3]; 
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

#define NAIVE_MAXPOOL2D_IMPLEMENTATION(dtype) \
void naive_max_pool2d_ ## dtype (const dtype *in,  \
        dtype *out, i32 x, i32 y, i32 s)           \
{                                                  \
  i32 ox = x / s;                                  \
  i32 oy = y / s;                                  \
  i32 i, j, k, l;                                  \
  const dtype *curr;                               \
  dtype v_max;                                     \
  for (i = 0; i < oy; ++i) {                       \
    for (j = 0; j < ox; ++j) {                     \
      v_max = *(in + s * i * x + s * j);           \
      for (k = 0; k < s; ++k) {                    \
        for (l = 0; l < s; ++l) {                  \
          curr = in + (s * i + k) * x + j * s + l; \
          v_max = *curr > v_max ? *curr : v_max;   \
        }                                          \
      }                                            \
      *out++ = v_max;                              \
    }                                              \
  }                                                \
}

NAIVE_MAXPOOL2D_IMPLEMENTATION  (i8);
NAIVE_MAXPOOL2D_IMPLEMENTATION  (u8);
NAIVE_MAXPOOL2D_IMPLEMENTATION  (i16);
NAIVE_MAXPOOL2D_IMPLEMENTATION  (u16);
NAIVE_MAXPOOL2D_IMPLEMENTATION  (i32);
NAIVE_MAXPOOL2D_IMPLEMENTATION  (u32);
NAIVE_MAXPOOL2D_IMPLEMENTATION  (i64);
NAIVE_MAXPOOL2D_IMPLEMENTATION  (u64);
NAIVE_MAXPOOL2D_IMPLEMENTATION  (f32);
NAIVE_MAXPOOL2D_IMPLEMENTATION  (f64);

#define DFL_MAXPOOL2D_IMPLEMENTATION(dtype) \
void ecpu_max_pool2d_ ## dtype (                          \
	const dtype *in, dtype *out, i32 x, i32 y, i32 s) \
{                                                         \
	naive_max_pool2d_ ## dtype(in, out, x, y, s);     \
}

DFL_MAXPOOL2D_IMPLEMENTATION  (i8);
DFL_MAXPOOL2D_IMPLEMENTATION  (u8);
DFL_MAXPOOL2D_IMPLEMENTATION  (i16);
DFL_MAXPOOL2D_IMPLEMENTATION  (u16);
DFL_MAXPOOL2D_IMPLEMENTATION  (i32);
DFL_MAXPOOL2D_IMPLEMENTATION  (u32);
DFL_MAXPOOL2D_IMPLEMENTATION  (i64);
DFL_MAXPOOL2D_IMPLEMENTATION  (u64);
/* DFL_MAXPOOL2D_IMPLEMENTATION  (f32); */
DFL_MAXPOOL2D_IMPLEMENTATION  (f64);


void ecpu_max_pool2d_f32(const f32 *in, f32 *out, i32 x, i32 y, i32 s)
{
	switch (s) {
#ifdef ALT_MAXPOOL2D_F32_S2
	case 2:
		ALT_MAXPOOL2D_F32_S2(in, out, x, y, s);
		break;
#endif
	default:
		naive_max_pool2d_f32(in, out, x, y, s);
		break;
	}
}

#if defined(__x86_64) && defined(__SSE__)
void sse_max_pool2d_f32_s2(const f32 *in, f32 *out, i32 x, i32 y, i32 s)
{
	i32 i, j;
	__m128 blka;
	__m128 blkb;
	f32 maxl, maxh, a, b;
	f32 *pa = (f32*)&blka, *pb = ((f32*)&blka) + 2;
	for (i = 0; i < (y - 1); i += 2) {
		for (j = 0; j < (x - 3); j += 4) {
			blka = _mm_loadu_ps(in + (i + 0) * x + j);
			blkb = _mm_loadu_ps(in + (i + 1) * x + j);
			blka = _mm_max_ps(blka, blkb);
			*out++ = pa[0] > pa[1] ? pa[0] : pa[1];  
			*out++ = pb[0] > pb[1] ? pb[0] : pb[1];  
		}
		for (; j < (x - 1); j += 2) {
			a = *(in + (i + 0) * x + j + 0);
			b = *(in + (i + 0) * x + j + 1);
			maxl = a > b ? a : b;
			a = *(in + (i + 1) * x + j + 0);
			b = *(in + (i + 1) * x + j + 1);
			maxh = a > b ? a : b;
			*out++ = maxl > maxh ? maxl : maxh;
		}
	}
}
#endif

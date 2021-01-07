#include <string.h>

#ifdef __x86_64
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
#include "sse2neon.h"
#endif

#include "ecpufn.h"

#define NAIVE_CONV2D_IMPLEMENTATION(dtype) \
void naive_conv2d_ ## dtype (const dtype *in,    \
    dtype *out, i32 x, i32 y, i32 sx, i32 sy,    \
    const dtype *k, i32 kw)                      \
{                                                \
  i32 i, j, u, v;                                \
  dtype acc;                                     \
  for (i = 0; i <= y - kw; i += sy) {            \
    for (j = 0; j <= x - kw; j += sx) {          \
      acc = 0;                                   \
      for (u = 0; u < kw; ++u) {                 \
        for (v = 0; v < kw; ++v) {               \
          acc += *(in + (i + u) * x + (j + v)) * \
                 *(k + u * kw + v);              \
        }                                        \
      }                                          \
      *out++ = acc;                              \
    }                                            \
  }                                              \
}

NAIVE_CONV2D_IMPLEMENTATION  (i8)
NAIVE_CONV2D_IMPLEMENTATION  (u8)
NAIVE_CONV2D_IMPLEMENTATION  (i16)
NAIVE_CONV2D_IMPLEMENTATION  (u16)
NAIVE_CONV2D_IMPLEMENTATION  (i32)
NAIVE_CONV2D_IMPLEMENTATION  (u32)
NAIVE_CONV2D_IMPLEMENTATION  (i64)
NAIVE_CONV2D_IMPLEMENTATION  (u64)
NAIVE_CONV2D_IMPLEMENTATION  (f32)
NAIVE_CONV2D_IMPLEMENTATION  (f64)

#define DFL_CONV2D_IMPLEMENTATION(dtype) \
void ecpu_conv2d_ ## dtype (const dtype *in, dtype *out,      \
	i32 x, i32 y, i32 sx, i32 sy, const dtype *k, i32 kw) \
{                                                             \
	naive_conv2d_ ## dtype(in, out, x, y, sx, sy, k, kw); \
}

DFL_CONV2D_IMPLEMENTATION  (i8)
DFL_CONV2D_IMPLEMENTATION  (u8)
DFL_CONV2D_IMPLEMENTATION  (i16)
DFL_CONV2D_IMPLEMENTATION  (u16)
DFL_CONV2D_IMPLEMENTATION  (i32)
DFL_CONV2D_IMPLEMENTATION  (u32)
DFL_CONV2D_IMPLEMENTATION  (i64)
DFL_CONV2D_IMPLEMENTATION  (u64)
/* DFL_CONV2D_IMPLEMENTATION  (f32) */
DFL_CONV2D_IMPLEMENTATION  (f64)

void ecpu_conv2d_f32(const f32 *in, f32 *out,
	i32 x, i32 y, i32 sx, i32 sy, const f32 *k, i32 kw)
{
	switch (kw) {
#ifdef ALT_CONV2D_F32_K1S1
	case 1:
		if (sx == sy && sx == 1)
			ALT_CONV2D_F32_K1S1(
				in, out, x, y, sx, sy, k, kw);
		else
			naive_conv2d_f32(
				in, out, x, y, sx, sy, k, kw);
		break;
#endif
#ifdef ALT_CONV2D_F32_K2SX
	case 2:
		ALT_CONV2D_F32_K2SX(in, out, x, y, sx, sy, k, kw);
		break;
#endif
#ifdef ALT_CONV2D_F32_K3SX
	case 3:
		if (sx == sy && sx == 1)
			ALT_CONV2D_F32_K3S1(
				in, out, x, y, sx, sy, k, kw);
		else
			ALT_CONV2D_F32_K3SX(
				in, out, x, y, sx, sy, k, kw);
		break;
#ifdef ALT_CONV2D_F32_K4SX
	case 4:
		ALT_CONV2D_F32_K4SX(in, out, x, y, sx, sy, k, kw);
		break;
#endif
#ifdef ALT_CONV2D_F32_K5SX
	case 5:
		ALT_CONV2D_F32_K5SX(in, out, x, y, sx, sy, k, kw);
		break;
#endif
#endif
	default:
		naive_conv2d_f32(in, out, x, y, sx, sy, k, kw);
		break;
	}
}

#if defined(__x86_64) && defined(__SSE__) || \
    defined(__ARM_NEON) /* Supported via `sse2neon` */
void sse_conv2d_f32_k1s1(const f32 *in, f32 *out,
	i32 x, i32 y, i32 sx, i32 sy, const f32 *k, i32 kw)
{
	i32 i, size, nblk;
	__m128 lvk, blk;
	lvk = _mm_set1_ps(*k);
	size = x * y;
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

void sse_conv2d_f32_k2sx(const f32 *in, f32 *out,
	i32 x, i32 y, i32 sx, i32 sy, const f32 *k, i32 kw)
{
	i32 i, j, off;
	__m128 lvk, blk;
	f32 *res = (f32*)&(blk);
	lvk = _mm_set_ps(k[3], k[2], k[1], k[0]);
	for (i = 0; i <= y - kw; i += sy) {
		for (j = 0; j <= x - kw; j += sx) {
			off = i * x + j;
			blk = _mm_shuffle_ps(
				_mm_loadu_ps(in + off),
				_mm_loadu_ps(in + off + x - 2),
				_MM_SHUFFLE(3, 2, 1, 0));
			blk = blk * lvk;
			*res += res[1] + res[2] + res[3];
			*out++ = *res;
		}
	}
}

void sse_conv2d_f32_k3s1(const f32 *in, f32 *out,
	i32 x, i32 y, i32 sx, i32 sy, const f32 *k, i32 kw)
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
	for (i = 0; i <= y - kw - sy; i += sy) {
		for (j = 0; j < x - kw; j += 2) {
			accl = _mm_setzero_ps();
			acch = _mm_setzero_ps();
			off = i * x + j;
			for (u = 0; u < kw; ++u) {
				blk = _mm_loadu_ps(in + off);
				accl += blk * vkl[u];
				acch += blk * vkh[u];
				off += x;
			}
			*out++ = (resl[0] + resl[1] + resl[2]);
			*out++ = (resh[1] + resh[2] + resh[3]);
		}
		if (j == x - kw) {
			accl = _mm_setzero_ps();
			off = i * x + j;
			for (u = 0; u < kw; ++u) {
				blk = _mm_loadu_ps(in + off);
				accl += blk * vkl[u];
				off  += x;
			}
			*out++ = (resl[0] + resl[1] + resl[2]);
		}
	}
	if (y - kw >= i) {
		off = i * x;
		for (j = -1; j <= x - kw - 1; ++j) {
			acch = _mm_setzero_ps();
			off = i * x + j;
			for (u = 0; u < kw; ++u) {
				blk = _mm_loadu_ps(in + off);
				acch += blk * vkh[u];
				off  += x;
			}
			*out++ = (resh[1] + resh[2] + resh[3]);
		}
	}
}

void sse_conv2d_f32_k3sx(const f32 *in, f32 *out,
	i32 x, i32 y, i32 sx, i32 sy, const f32 *k, i32 kw)
{
	i32 i, j, u, off;
	__m128 lvk[3];
	__m128 blk, acc;
	f32 *res = (f32*)&(acc);
	for (u = 0; u < kw; ++u) {
		memcpy(&(lvk[u]), k + u * kw, sizeof(f32) * kw);
	}
	for (i = 0; i <= y - kw - sy; i += sy) {
		for (j = 0; j <= x - kw; j += sx) {
			acc = _mm_setzero_ps();
			off = i * x + j;
			for (u = 0; u < kw; ++u) {
				blk = _mm_loadu_ps(in + off);
				acc += blk * lvk[u];
				off += x;
			}
			*out++ = (res[0] + res[1] + res[2]);
		}
	}
	if (y - kw >= i) {
		off = i * x;
		for (u = 0; u < kw; ++u) {
			memcpy(((f32*)&(lvk[u])) + 1,
				k + u * kw, sizeof(f32) * kw);
		}
		for (j = -1; j <= x - kw - 1; j += sx) {
			acc = _mm_setzero_ps();
			off = i * x + j;
			for (u = 0; u < kw; ++u) {
				blk = _mm_loadu_ps(in + off);
				acc += blk * lvk[u];
				off += x;
			}
			*out++ = (res[1] + res[2] + res[3]);
		}
	}
}

void sse_conv2d_f32_k4sx(const f32 *in, f32 *out,
	i32 x, i32 y, i32 sx, i32 sy, const f32 *k, i32 kw)
{
	i32 i, j, u, off;
	__m128 lvk[4];
	__m128 blk, acc;
	f32 *res = (f32*)&(acc);
	for (u = 0; u < kw; ++u) {
		memcpy(&(lvk[u]), k + u * kw, sizeof(f32) * kw);
	}
	for (i = 0; i <= y - kw; i += sy) {
		for (j = 0; j <= x - kw; j += sx) {
			acc = _mm_setzero_ps();
			off = i * x + j;
			for (u = 0; u < kw; ++u) {
				blk = _mm_loadu_ps(in + off);
				acc += blk * lvk[u];
				off += x;
			}
			*out++ = res[0] + res[1] + res[2] + res[3];
		}
	}
}

void sse_conv2d_f32_k5sx(const f32 *in, f32 *out,
	i32 x, i32 y, i32 sx, i32 sy, const f32 *k, i32 kw)
{
	i32 i, j, u, off;
	__m128 vk4[5];
	__m128 blkv, accv;
	f32 blks, accs, sk5[5];
	f32 *res = (f32*)&(accv);
	for (u = 0; u < kw; ++u) {
		memcpy(&(vk4[u]), k + u * kw, sizeof(f32) * 4);
		sk5[u] = *(k + u * kw + 4);
	}
	for (i = 0; i <= y - kw; i += sy) {
		for (j = 0; j <= x - kw; j += sx) {
			accv = _mm_setzero_ps();
			accs = 0;
			off = i * x + j;
			for (u = 0; u < kw; ++u) {
				blkv = _mm_loadu_ps(in + off);
				accv += blkv * vk4[u];
				blks = *(in + off + 4);
				accs += blks * sk5[u];
				off += x;
			}
			*out++ = res[0] + res[1] + res[2] + res[3] + accs;
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

NAIVE_DOTPROD_IMPLEMENTATION  (i8)
NAIVE_DOTPROD_IMPLEMENTATION  (u8)
NAIVE_DOTPROD_IMPLEMENTATION  (i16)
NAIVE_DOTPROD_IMPLEMENTATION  (u16)
NAIVE_DOTPROD_IMPLEMENTATION  (i32)
NAIVE_DOTPROD_IMPLEMENTATION  (u32)
NAIVE_DOTPROD_IMPLEMENTATION  (i64)
NAIVE_DOTPROD_IMPLEMENTATION  (u64)
NAIVE_DOTPROD_IMPLEMENTATION  (f32)
NAIVE_DOTPROD_IMPLEMENTATION  (f64)

#define DFL_DOTPROD_IMPLEMENTATION(dtype) \
void ecpu_dot_prod_ ## dtype (const dtype *in,     \
	dtype *out, const dtype *w, i32 iw)        \
{                                                  \
	naive_dot_prod_ ## dtype (in, out, w, iw); \
}

DFL_DOTPROD_IMPLEMENTATION  (i8)
DFL_DOTPROD_IMPLEMENTATION  (u8)
DFL_DOTPROD_IMPLEMENTATION  (i16)
DFL_DOTPROD_IMPLEMENTATION  (u16)
DFL_DOTPROD_IMPLEMENTATION  (i32)
DFL_DOTPROD_IMPLEMENTATION  (u32)
DFL_DOTPROD_IMPLEMENTATION  (i64)
DFL_DOTPROD_IMPLEMENTATION  (u64)
/* DFL_DOTPROD_IMPLEMENTATION  (f32) */
DFL_DOTPROD_IMPLEMENTATION  (f64)

void ecpu_dot_prod_f32(const f32 *in, f32 *out, const f32 *w, i32 iw)
{
#ifdef ALT_DOTPROD_F32
	ALT_DOTPROD_F32(in, out, w, iw);
#else
	naive_dot_prod_f32(in, out, w, iw);
#endif
}

#if defined(__x86_64) && defined(__SSE__) || \
    defined(__ARM_NEON) /* Supported via `sse2neon` */
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
void naive_max_pool2d_ ## dtype (const dtype *in,       \
    dtype *out, i32 x, i32 y, i32 sx, i32 sy, i32 kw)   \
{                                                       \
  i32 i, j, u, v;                                       \
  dtype max;                                            \
  for (i = 0; i <= y - kw; i += sy) {                   \
    for (j = 0; j <= x - kw; j += sx) {                 \
      max = *(in + i * x + j);                          \
        for (u = 0; u < kw; ++u) {                      \
          for (v = 0; v < kw; ++v) {                    \
            max = max < *(in + (i + u) * x + (j + v)) ? \
                   *(in + (i + u) * x + (j + v)) : max; \
          }                                             \
        }                                               \
        *out++ = max;                                   \
    }                                                   \
  }                                                     \
}

NAIVE_MAXPOOL2D_IMPLEMENTATION  (i8)
NAIVE_MAXPOOL2D_IMPLEMENTATION  (u8)
NAIVE_MAXPOOL2D_IMPLEMENTATION  (i16)
NAIVE_MAXPOOL2D_IMPLEMENTATION  (u16)
NAIVE_MAXPOOL2D_IMPLEMENTATION  (i32)
NAIVE_MAXPOOL2D_IMPLEMENTATION  (u32)
NAIVE_MAXPOOL2D_IMPLEMENTATION  (i64)
NAIVE_MAXPOOL2D_IMPLEMENTATION  (u64)
NAIVE_MAXPOOL2D_IMPLEMENTATION  (f32)
NAIVE_MAXPOOL2D_IMPLEMENTATION  (f64)

#define DFL_MAXPOOL2D_IMPLEMENTATION(dtype) \
void ecpu_max_pool2d_ ## dtype (const dtype *in,               \
	dtype *out, i32 x, i32 y, i32 sx, i32 sy, i32 kw)      \
{                                                              \
	naive_max_pool2d_ ## dtype(in, out, x, y, sx, sy, kw); \
}

DFL_MAXPOOL2D_IMPLEMENTATION  (i8)
DFL_MAXPOOL2D_IMPLEMENTATION  (u8)
DFL_MAXPOOL2D_IMPLEMENTATION  (i16)
DFL_MAXPOOL2D_IMPLEMENTATION  (u16)
DFL_MAXPOOL2D_IMPLEMENTATION  (i32)
DFL_MAXPOOL2D_IMPLEMENTATION  (u32)
DFL_MAXPOOL2D_IMPLEMENTATION  (i64)
DFL_MAXPOOL2D_IMPLEMENTATION  (u64)
/* DFL_MAXPOOL2D_IMPLEMENTATION  (f32) */
DFL_MAXPOOL2D_IMPLEMENTATION  (f64)

void ecpu_max_pool2d_f32(const f32 *in, f32 *out,
	i32 x, i32 y, i32 sx, i32 sy, i32 kw)
{
	switch (kw) {
#ifdef ALT_MAXPOOL2D_F32_K2S2
	case 2:
		if (sx == sy && sx == 2)
			ALT_MAXPOOL2D_F32_K2S2(in, out, x, y, sx, sy, kw);
		else
			naive_max_pool2d_f32(in, out, x, y, sx, sy, kw);
		break;
#endif
#ifdef ALT_MAXPOOL2D_F32_K3S3
	case 3:
		if (sx == sy && sx == 3)
			ALT_MAXPOOL2D_F32_K3S3(in, out, x, y, sx, sy, kw);
		else
			naive_max_pool2d_f32(in, out, x, y, sx, sy, kw);
		break;
#endif
#ifdef ALT_MAXPOOL2D_F32_K4S4
	case 4:
		if (sx == sy && sx == 4)
			ALT_MAXPOOL2D_F32_K4S4(in, out, x, y, sx, sy, kw);
		else
			naive_max_pool2d_f32(in, out, x, y, sx, sy, kw);
		break;
#endif
	default:
		naive_max_pool2d_f32(in, out, x, y, sx, sy, kw);
		break;
	}
}

#if defined(__x86_64) && defined(__SSE__) || \
    defined(__ARM_NEON) /* Supported via `sse2neon` */
void sse_max_pool2d_f32_k2s2(const f32 *in,
	f32 *out, i32 x, i32 y, i32 sx, i32 sy, i32 kw)
{
	i32 i, j;
	__m128 blka;
	__m128 blkb;
	f32 maxl, maxh, a, b;
	f32 *pa = (f32*)&blka, *pb = ((f32*)&blka) + 2;
	for (i = 0; i <= (y - 2); i += 2) {
		for (j = 0; j <= (x - 4); j += 4) {
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

void sse_max_pool2d_f32_k3s3(const f32 *in,
	f32 *out, i32 x, i32 y, i32 sx, i32 sy, i32 kw)
{
	i32 i, j;
	__m128 blka, blkb, blkc;
	f32 t;
	f32 *p = (f32*)&blka;
	for (i = 0; i <= (y - 3); i += 3) {
		for (j = 0; j <= (x - 4); j += 3) {
			blka = _mm_loadu_ps(in + (i + 0) * x + j);
			blkb = _mm_loadu_ps(in + (i + 1) * x + j);
			blkc = _mm_loadu_ps(in + (i + 2) * x + j);
			blka = _mm_max_ps(blka, blkb);
			blka = _mm_max_ps(blka, blkc);
			t = p[0] > p[1] ? p[0] : p[1];
			*out++ = t > p[2] ? t : p[2];
		}
		if (j == x - 3) {
			blka = _mm_loadu_ps(in + (i + 0) * x + j - 1);
			blkb = _mm_loadu_ps(in + (i + 1) * x + j - 1);
			blkc = _mm_loadu_ps(in + (i + 2) * x + j - 1);
			blka = _mm_max_ps(blka, blkb);
			blka = _mm_max_ps(blka, blkc);
			t = p[1] > p[2] ? p[1] : p[2];
			*out++ = t > p[3] ? t : p[3];
		}
	}
}

void sse_max_pool2d_f32_k4s4(const f32 *in,
	f32 *out, i32 x, i32 y, i32 sx, i32 sy, i32 kw)
{
	i32 i, j;
	__m128 blka, blkb, blkc, blkd;
	f32 maxh, maxl;
	f32 *p = (f32*)&blka;
	for (i = 0; i <= (y - 4); i += 4) {
		for (j = 0; j <= (x - 4); j += 4) {
			blka = _mm_loadu_ps(in + (i + 0) * x + j);
			blkb = _mm_loadu_ps(in + (i + 1) * x + j);
			blkc = _mm_loadu_ps(in + (i + 2) * x + j);
			blkd = _mm_loadu_ps(in + (i + 3) * x + j);
			blka = _mm_max_ps(blka, blkb);
			blkc = _mm_max_ps(blkc, blkd);
			blka = _mm_max_ps(blka, blkc);
			maxl = p[0] > p[1] ? p[0] : p[1];
			maxh = p[2] > p[3] ? p[2] : p[3];
			*out++ = maxh > maxl ? maxh : maxl;
		}
	}
}
#endif

#define NAIVE_AVGPOOL2D_IMPLEMENTATION(dtype) \
void naive_avg_pool2d_ ## dtype (const dtype *in,     \
    dtype *out, i32 x, i32 y, i32 sx, i32 sy, i32 kw) \
{                                                     \
  i32 i, j, u, v, n = kw * kw;                        \
  dtype avg;                                          \
  for (i = 0; i <= y - kw; i += sy) {                 \
    for (j = 0; j <= x - kw; j += sx) {               \
      avg = 0;                                        \
      for (u = 0; u < kw; ++u) {                      \
        for (v = 0; v < kw; ++v) {                    \
          avg += *(in + (i + u) * x + (j + v));       \
        }                                             \
      }                                               \
      *out++ = avg / n;                               \
    }                                                 \
  }                                                   \
}

NAIVE_AVGPOOL2D_IMPLEMENTATION  (i8)
NAIVE_AVGPOOL2D_IMPLEMENTATION  (u8)
NAIVE_AVGPOOL2D_IMPLEMENTATION  (i16)
NAIVE_AVGPOOL2D_IMPLEMENTATION  (u16)
NAIVE_AVGPOOL2D_IMPLEMENTATION  (i32)
NAIVE_AVGPOOL2D_IMPLEMENTATION  (u32)
NAIVE_AVGPOOL2D_IMPLEMENTATION  (i64)
NAIVE_AVGPOOL2D_IMPLEMENTATION  (u64)
NAIVE_AVGPOOL2D_IMPLEMENTATION  (f32)
NAIVE_AVGPOOL2D_IMPLEMENTATION  (f64)

#define DFL_AVGPOOL2D_IMPLEMENTATION(dtype) \
void ecpu_avg_pool2d_ ## dtype (const dtype *in,               \
	dtype *out, i32 x, i32 y, i32 sx, i32 sy, i32 kw)      \
{                                                              \
	naive_avg_pool2d_ ## dtype(in, out, x, y, sx, sy, kw); \
}

DFL_AVGPOOL2D_IMPLEMENTATION  (i8)
DFL_AVGPOOL2D_IMPLEMENTATION  (u8)
DFL_AVGPOOL2D_IMPLEMENTATION  (i16)
DFL_AVGPOOL2D_IMPLEMENTATION  (u16)
DFL_AVGPOOL2D_IMPLEMENTATION  (i32)
DFL_AVGPOOL2D_IMPLEMENTATION  (u32)
DFL_AVGPOOL2D_IMPLEMENTATION  (i64)
DFL_AVGPOOL2D_IMPLEMENTATION  (u64)
/* DFL_AVGPOOL2D_IMPLEMENTATION  (f32) */
DFL_AVGPOOL2D_IMPLEMENTATION  (f64)

void ecpu_avg_pool2d_f32(const f32 *in,
	f32 *out, i32 x, i32 y, i32 sx, i32 sy, i32 kw)
{
	switch (kw) {
#ifdef ALT_AVGPOOL2D_F32_K2S2
	case 2:
		if (sx == sy && sx == 2)
			ALT_AVGPOOL2D_F32_K2S2(in, out, x, y, sx, sy, kw);
		else
			naive_avg_pool2d_f32(in, out, x, y, sx, sy, kw);
		break;
#endif
#ifdef ALT_AVGPOOL2D_F32_K3S3
	case 3:
		if (sx == sy && sx == 3)
			ALT_AVGPOOL2D_F32_K3S3(in, out, x, y, sx, sy, kw);
		else
			naive_avg_pool2d_f32(in, out, x, y, sx, sy, kw);
		break;
#endif
#ifdef ALT_AVGPOOL2D_F32_K4S4
	case 4:
		if (sx == sy && sx == 4)
			ALT_AVGPOOL2D_F32_K4S4(in, out, x, y, sx, sy, kw);
		else
			naive_avg_pool2d_f32(in, out, x, y, sx, sy, kw);
		break;
#endif
	default:
		naive_avg_pool2d_f32(in, out, x, y, sx, sy, kw);
		break;
	}
}

#if defined(__x86_64) && defined(__SSE__) || \
    defined(__ARM_NEON) /* Supported via `sse2neon` */
void sse_avg_pool2d_f32_k2s2(const f32 *in,
	f32 *out, i32 x, i32 y, i32 sx, i32 sy, i32 kw)
{
	i32 i, j;
	__m128 blka, blkb;
	f32 sum;
	f32 *pa = (f32*)&blka, *pb = ((f32*)&blka) + 2;
	for (i = 0; i <= (y - 2); i += 2) {
		for (j = 0; j <= (x - 4); j += 4) {
			blka = _mm_loadu_ps(in + (i + 0) * x + j);
			blkb = _mm_loadu_ps(in + (i + 1) * x + j);
			blka = blka + blkb;
			*out++ = (pa[0] + pa[1]) / 4;
			*out++ = (pb[0] + pb[1]) / 4;
		}
		for (; j < (x - 1); j += 2) {
			sum  = *(in + (i + 0) * x + j + 0);
			sum += *(in + (i + 0) * x + j + 1);
			sum += *(in + (i + 1) * x + j + 0);
			sum += *(in + (i + 1) * x + j + 1);
			*out++ = sum / 4;
		}
	}
}

void sse_avg_pool2d_f32_k3s3(const f32 *in,
	f32 *out, i32 x, i32 y, i32 sx, i32 sy, i32 kw)
{
	i32 i, j;
	__m128 blka, blkb, blkc;
	f32 *p = (f32*)&blka;
	for (i = 0; i <= (y - 3); i += 3) {
		for (j = 0; j <= (x - 4); j += 3) {
			blka = _mm_loadu_ps(in + (i + 0) * x + j);
			blkb = _mm_loadu_ps(in + (i + 1) * x + j);
			blkc = _mm_loadu_ps(in + (i + 2) * x + j);
			blka = blka + blkb + blkc;
			*out++ = (p[0] + p[1] + p[2]) / 9;
		}
		if (j == x - 3) {
			blka = _mm_loadu_ps(in + (i + 0) * x + j - 1);
			blkb = _mm_loadu_ps(in + (i + 1) * x + j - 1);
			blkc = _mm_loadu_ps(in + (i + 2) * x + j - 1);
			blka = blka + blkb + blkc;
			*out++ = (p[1] + p[2] + p[3]) / 9;
		}
	}
}

void sse_avg_pool2d_f32_k4s4(const f32 *in,
	f32 *out, i32 x, i32 y, i32 sx, i32 sy, i32 kw)
{
	i32 i, j;
	__m128 blka, blkb, blkc, blkd;
	f32 *p = (f32*)&blka;
	for (i = 0; i <= (y - 4); i += 4) {
		for (j = 0; j <= (x - 4); j += 4) {
			blka = _mm_loadu_ps(in + (i + 0) * x + j);
			blkb = _mm_loadu_ps(in + (i + 1) * x + j);
			blkc = _mm_loadu_ps(in + (i + 2) * x + j);
			blkd = _mm_loadu_ps(in + (i + 3) * x + j);
			blka = blka + blkb + blkc + blkd;
			*out++ = (p[0] + p[1] + p[2] + p[3]) / 16;
		}
	}
}
#endif

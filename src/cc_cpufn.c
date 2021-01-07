#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cc_assert.h"
#include "util_log.h"
#include "cc_cpufn.h"

#define UNSUPPORTED_DTYPE_LOG(dt) \
utlog_format(UTLOG_ERR,                     \
"cc_cpufn: unsupported dtype [%x]@%s: %d\n",\
dt, __FILE__, __LINE__);

#define CC_CPU_RELU_CASE_TEMP(_dt) \
for (i = 0; i < elems; ++i) {            \
	*(((_dt*)inp) + i) =             \
		*(((_dt*)inp) + i) > 0 ? \
		*(((_dt*)inp) + i) : 0;  \
}

void cc_cpu_activation_relu(void *inp, cc_int32 elems, cc_dtype dt)
{
	cc_int32 i;
	switch (dt) {
	case CC_UINT8:
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
		CC_CPU_RELU_CASE_TEMP(cc_uint8);
		break;
	case CC_UINT16:
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
		CC_CPU_RELU_CASE_TEMP(cc_uint16);
		break;
	case CC_UINT32:
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
		CC_CPU_RELU_CASE_TEMP(cc_uint32);
		break;
	case CC_UINT64:
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
		CC_CPU_RELU_CASE_TEMP(cc_uint64);
		break;
	case CC_INT8:
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
		CC_CPU_RELU_CASE_TEMP(cc_int8);
		break;
	case CC_INT16:
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
		CC_CPU_RELU_CASE_TEMP(cc_int16);
		break;
	case CC_INT32:
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
		CC_CPU_RELU_CASE_TEMP(cc_int32);
		break;
	case CC_INT64:
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
		CC_CPU_RELU_CASE_TEMP(cc_int64);
		break;
	case CC_FLOAT32:
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
		CC_CPU_RELU_CASE_TEMP(cc_float32);
		break;
	case CC_FLOAT64:
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
		CC_CPU_RELU_CASE_TEMP(cc_float64);
		break;
	default:
		UNSUPPORTED_DTYPE_LOG(dt);
		break;
	}
}

#define CC_CPU_RELU6_CASE_TEMP(_dt) \
for (i = 0; i < elems; ++i) {            \
	*(((_dt*)inp) + i) =             \
		*(((_dt*)inp) + i) > 0 ? \
		*(((_dt*)inp) + i) : 0;  \
	*(((_dt*)inp) + i) =             \
		*(((_dt*)inp) + i) > 6 ? \
		6 : *(((_dt*)inp) + i);  \
}

void cc_cpu_activation_relu6(void *inp, cc_int32 elems, cc_dtype dt)
{
	cc_int32 i;
	switch (dt) {
	case CC_UINT8:
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
		CC_CPU_RELU6_CASE_TEMP(cc_uint8);
		break;
	case CC_UINT16:
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
		CC_CPU_RELU6_CASE_TEMP(cc_uint16);
		break;
	case CC_UINT32:
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
		CC_CPU_RELU6_CASE_TEMP(cc_uint32);
		break;
	case CC_UINT64:
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
		CC_CPU_RELU6_CASE_TEMP(cc_uint64);
		break;
	case CC_INT8:
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
		CC_CPU_RELU6_CASE_TEMP(cc_int8);
		break;
	case CC_INT16:
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
		CC_CPU_RELU6_CASE_TEMP(cc_int16);
		break;
	case CC_INT32:
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
		CC_CPU_RELU6_CASE_TEMP(cc_int32);
		break;
	case CC_INT64:
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
		CC_CPU_RELU6_CASE_TEMP(cc_int64);
		break;
	case CC_FLOAT32:
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
		CC_CPU_RELU6_CASE_TEMP(cc_float32);
		break;
	case CC_FLOAT64:
#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(i)
#endif
		CC_CPU_RELU6_CASE_TEMP(cc_float64);
		break;
	default:
		UNSUPPORTED_DTYPE_LOG(dt);
		break;
	}
}

static void cc_cpu_activation_softmax_float32(
	cc_float32 *inp, cc_int32 elems)
{
	cc_int32 i;
	cc_float32 v, s = 0;
	for (i = 0; i < elems; ++i) {
#ifdef CONFIG_STD_C89
		v = (cc_float32)exp(inp[i]);
		inp[i] = v;
		s += v;
#else
		v = expf(inp[i]);
		inp[i] = v;
		s += v;
#endif
	}
	for (i = 0; i < elems; ++i) {
		inp[i] = inp[i] / s;
	}
}

static void cc_cpu_activation_softmax_float64(
	cc_float64 *inp, cc_int32 elems)
{
	cc_int32 i;
	cc_float64 v, s = 0;
	for (i = 0; i < elems; ++i) {
		v = exp(inp[i]);
		inp[i] = v;
		s += v;
	}
	for (i = 0; i < elems; ++i) {
		inp[i] = inp[i] / s;
	}
}

void cc_cpu_activation_softmax(void *inp, cc_int32 elems, cc_dtype dt)
{
	switch (dt) {
	case CC_FLOAT32:
		cc_cpu_activation_softmax_float32(
			(cc_float32*)inp, elems);
		break;
	case CC_FLOAT64:
		cc_cpu_activation_softmax_float64(
			(cc_float64*)inp, elems);
		break;
	default:
		UNSUPPORTED_DTYPE_LOG(dt);
		break;
	}
}

#define CC_CPU_MAX_POO2D_IMPLEMENTATION(dt) \
static void cc_cpu_max_pool2d_ ## dt (const cc_ ## dt *in, \
  cc_ ## dt *out, cc_int32 x, cc_int32 y, cc_int32 sx,     \
 cc_int32 sy, cc_int32 kw)                                 \
{                                                          \
  cc_int32 i, j, u, v;                                     \
  cc_ ## dt max;                                           \
  for (i = 0; i <= y - kw; i += sy) {                      \
    for (j = 0; j <= x - kw; j += sx) {                    \
      max = *(in + i * x + j);                             \
        for (u = 0; u < kw; ++u) {                         \
          for (v = 0; v < kw; ++v) {                       \
            max = max < *(in + (i + u) * x + (j + v)) ?    \
                   *(in + (i + u) * x + (j + v)) : max;    \
          }                                                \
        }                                                  \
        *out++ = max;                                      \
    }                                                      \
  }                                                        \
}

CC_CPU_MAX_POO2D_IMPLEMENTATION  (uint8)
CC_CPU_MAX_POO2D_IMPLEMENTATION  (uint16)
CC_CPU_MAX_POO2D_IMPLEMENTATION  (uint32)
CC_CPU_MAX_POO2D_IMPLEMENTATION  (uint64)
CC_CPU_MAX_POO2D_IMPLEMENTATION  (int8)
CC_CPU_MAX_POO2D_IMPLEMENTATION  (int16)
CC_CPU_MAX_POO2D_IMPLEMENTATION  (int32)
CC_CPU_MAX_POO2D_IMPLEMENTATION  (int64)
CC_CPU_MAX_POO2D_IMPLEMENTATION  (float32)
CC_CPU_MAX_POO2D_IMPLEMENTATION  (float64)

#define CC_CPU_MAX_POOL2D_CASE(DT, dtype) \
case DT:                                                      \
	cc_cpu_max_pool2d_ ## dtype((const cc_ ## dtype*)inp, \
		(cc_ ## dtype*)oup, x, y, sx, sy, kw);        \
	break;
void cc_cpu_max_pool2d(const void *inp, void *oup, cc_int32 x,
	cc_int32 y, cc_int32 sx, cc_int32 sy, cc_int32 kw, cc_dtype dt)
{
	switch (dt) {
		CC_CPU_MAX_POOL2D_CASE(CC_UINT8,   uint8)
		CC_CPU_MAX_POOL2D_CASE(CC_UINT16,  uint16)
		CC_CPU_MAX_POOL2D_CASE(CC_UINT32,  uint32)
		CC_CPU_MAX_POOL2D_CASE(CC_UINT64,  uint64)
		CC_CPU_MAX_POOL2D_CASE(CC_INT8,    int8)
		CC_CPU_MAX_POOL2D_CASE(CC_INT16,   int16)
		CC_CPU_MAX_POOL2D_CASE(CC_INT32,   int32)
		CC_CPU_MAX_POOL2D_CASE(CC_INT64,   int64)
		CC_CPU_MAX_POOL2D_CASE(CC_FLOAT32, float32)
		CC_CPU_MAX_POOL2D_CASE(CC_FLOAT64, float64)
	default:
		UNSUPPORTED_DTYPE_LOG(dt);
		break;
	}
}

#define CC_CPU_AVG_POO2D_IMPLEMENTATION(dt) \
static void cc_cpu_avg_pool2d_ ## dt (const cc_ ## dt *in, \
  cc_ ## dt *out, cc_int32 ix, cc_int32 iy, cc_int32 sx,   \
 cc_int32 sy, cc_int32 kw)                                 \
{                                                          \
  cc_int32 i, j, u, v, n = kw * kw;                        \
  cc_ ## dt avg;                                           \
  for (i = 0; i <= iy - kw; i += sy) {                     \
    for (j = 0; j <= ix - kw; j += sx) {                   \
      avg = 0;                                             \
      for (u = 0; u < kw; ++u) {                           \
        for (v = 0; v < kw; ++v) {                         \
          avg += *(in + (i + u) * ix + (j + v));           \
        }                                                  \
      }                                                    \
      *out++ = avg / n;                                    \
    }                                                      \
  }                                                        \
}

CC_CPU_AVG_POO2D_IMPLEMENTATION  (uint8)
CC_CPU_AVG_POO2D_IMPLEMENTATION  (uint16)
CC_CPU_AVG_POO2D_IMPLEMENTATION  (uint32)
CC_CPU_AVG_POO2D_IMPLEMENTATION  (uint64)
CC_CPU_AVG_POO2D_IMPLEMENTATION  (int8)
CC_CPU_AVG_POO2D_IMPLEMENTATION  (int16)
CC_CPU_AVG_POO2D_IMPLEMENTATION  (int32)
CC_CPU_AVG_POO2D_IMPLEMENTATION  (int64)
CC_CPU_AVG_POO2D_IMPLEMENTATION  (float32)
CC_CPU_AVG_POO2D_IMPLEMENTATION  (float64)

#define CC_CPU_AVG_POOL2D_CASE(DT, dtype) \
case DT:                                                      \
	cc_cpu_avg_pool2d_ ## dtype((const cc_ ## dtype*)inp, \
		(cc_ ## dtype*)oup, x, y, sx, sy, kw);        \
	break;
void cc_cpu_avg_pool2d(const void *inp, void *oup, cc_int32 x,
	cc_int32 y, cc_int32 sx, cc_int32 sy, cc_int32 kw, cc_dtype dt)
{
	switch (dt) {
		CC_CPU_AVG_POOL2D_CASE(CC_UINT8,   uint8)
		CC_CPU_AVG_POOL2D_CASE(CC_UINT16,  uint16)
		CC_CPU_AVG_POOL2D_CASE(CC_UINT32,  uint32)
		CC_CPU_AVG_POOL2D_CASE(CC_UINT64,  uint64)
		CC_CPU_AVG_POOL2D_CASE(CC_INT8,    int8)
		CC_CPU_AVG_POOL2D_CASE(CC_INT16,   int16)
		CC_CPU_AVG_POOL2D_CASE(CC_INT32,   int32)
		CC_CPU_AVG_POOL2D_CASE(CC_INT64,   int64)
		CC_CPU_AVG_POOL2D_CASE(CC_FLOAT32, float32)
		CC_CPU_AVG_POOL2D_CASE(CC_FLOAT64, float64)
	default:
		UNSUPPORTED_DTYPE_LOG(dt);
		break;
	}
}

#define CC_CPU_CONV2D_IMPLEMENTATION(dt) \
static void cc_cpu_conv2d_ ## dt (const cc_ ## dt *in, \
  cc_ ## dt *out, cc_int32 x, cc_int32 y, cc_int32 sx, \
  cc_int32 sy, const cc_ ## dt *k, cc_int32 kw) {      \
  cc_int32 i, j, u, v;                                 \
  cc_ ## dt acc;                                       \
  for (i = 0; i <= y - kw; i += sy) {                  \
    for (j = 0; j <= x - kw; j += sx) {                \
      acc = 0;                                         \
      for (u = 0; u < kw; ++u) {                       \
        for (v = 0; v < kw; ++v) {                     \
          acc += *(in + (i + u) * x + (j + v)) *       \
                 *(k + u * kw + v);                    \
        }                                              \
      }                                                \
      *out++ = acc;                                    \
    }                                                  \
  }                                                    \
}

CC_CPU_CONV2D_IMPLEMENTATION  (uint8)
CC_CPU_CONV2D_IMPLEMENTATION  (uint16)
CC_CPU_CONV2D_IMPLEMENTATION  (uint32)
CC_CPU_CONV2D_IMPLEMENTATION  (uint64)
CC_CPU_CONV2D_IMPLEMENTATION  (int8)
CC_CPU_CONV2D_IMPLEMENTATION  (int16)
CC_CPU_CONV2D_IMPLEMENTATION  (int32)
CC_CPU_CONV2D_IMPLEMENTATION  (int64)
CC_CPU_CONV2D_IMPLEMENTATION  (float32)
CC_CPU_CONV2D_IMPLEMENTATION  (float64)

#define CC_CPU_CONV2D_CASE(DT, dtype) \
case DT:                                                              \
	cc_cpu_conv2d_ ## dtype((const cc_ ## dtype*)inp,             \
	(cc_ ## dtype*)oup, x, y, sx, sy, (cc_ ## dtype*)filter, fw); \
	break;
void cc_cpu_conv2d(const void *inp, void *oup,
	cc_int32 x, cc_int32 y, cc_int32 sx, cc_int32 sy,
	const void *filter, cc_int32 fw, cc_dtype dt)
{
	switch (dt) {
		CC_CPU_CONV2D_CASE(CC_UINT8,   uint8)
		CC_CPU_CONV2D_CASE(CC_UINT16,  uint16)
		CC_CPU_CONV2D_CASE(CC_UINT32,  uint32)
		CC_CPU_CONV2D_CASE(CC_UINT64,  uint64)
		CC_CPU_CONV2D_CASE(CC_INT8,    int8)
		CC_CPU_CONV2D_CASE(CC_INT16,   int16)
		CC_CPU_CONV2D_CASE(CC_INT32,   int32)
		CC_CPU_CONV2D_CASE(CC_INT64,   int64)
		CC_CPU_CONV2D_CASE(CC_FLOAT32, float32)
		CC_CPU_CONV2D_CASE(CC_FLOAT64, float64)
	default:
		UNSUPPORTED_DTYPE_LOG(dt);
		break;
	}
}

#ifndef CC_BN_OFFSET_CFG
#define CC_BN_OFFSET_CFG
enum {
	CC_BN_OFFSET_GAMMA = 0,
	CC_BN_OFFSET_BETA,
	CC_BN_OFFSET_MEAN,
	CC_BN_OFFSET_VAR,
	CC_BN_OFFSET_EPSILON,
	CC_BN_PARAMETERS
};
#endif

#define CC_CPU_BATCH_NORM_IMPLEMENTATION(dt) \
void cc_cpu_batch_norm_ ## dt(cc_ ## dt *inp,                     \
	cc_int32 len, cc_ ## dt *bnpara)                          \
{                                                                 \
	cc_ ## dt gamma = *(bnpara + CC_BN_OFFSET_GAMMA),         \
		beta    = *(bnpara + CC_BN_OFFSET_BETA),          \
		mean    = *(bnpara + CC_BN_OFFSET_MEAN),          \
		var     = *(bnpara + CC_BN_OFFSET_VAR),           \
		epsilon = *(bnpara + CC_BN_OFFSET_EPSILON);       \
	cc_ ## dt frac  = (cc_ ## dt)sqrt((double)var + epsilon); \
	cc_int32 i;                                               \
	for (i = 0; i < len; ++i) {                               \
		*(inp + i) = (gamma *                             \
				(*(inp + i) - mean) / frac)       \
			+ beta;                                   \
	}                                                         \
	return;                                                   \
}

CC_CPU_BATCH_NORM_IMPLEMENTATION  (uint8)
CC_CPU_BATCH_NORM_IMPLEMENTATION  (uint16)
CC_CPU_BATCH_NORM_IMPLEMENTATION  (uint32)
CC_CPU_BATCH_NORM_IMPLEMENTATION  (uint64)
CC_CPU_BATCH_NORM_IMPLEMENTATION  (int8)
CC_CPU_BATCH_NORM_IMPLEMENTATION  (int16)
CC_CPU_BATCH_NORM_IMPLEMENTATION  (int32)
CC_CPU_BATCH_NORM_IMPLEMENTATION  (int64)
CC_CPU_BATCH_NORM_IMPLEMENTATION  (float32)
CC_CPU_BATCH_NORM_IMPLEMENTATION  (float64)

#define CC_CPU_BATCH_NORM_CASE(DT, dtype) \
case DT:                                                 \
	cc_cpu_batch_norm_ ## dtype(                     \
	(cc_ ## dtype*)inp, len, (cc_ ## dtype*)bnpara); \
		break;
void cc_cpu_batch_norm(void *inp,
	cc_int32 len, const void *bnpara, cc_dtype dt)
{
	switch (dt) {
		CC_CPU_BATCH_NORM_CASE(CC_UINT8,   uint8)
		CC_CPU_BATCH_NORM_CASE(CC_UINT16,  uint16)
		CC_CPU_BATCH_NORM_CASE(CC_UINT32,  uint32)
		CC_CPU_BATCH_NORM_CASE(CC_UINT64,  uint64)
		CC_CPU_BATCH_NORM_CASE(CC_INT8,    int8)
		CC_CPU_BATCH_NORM_CASE(CC_INT16,   int16)
		CC_CPU_BATCH_NORM_CASE(CC_INT32,   int32)
		CC_CPU_BATCH_NORM_CASE(CC_INT64,   int64)
		CC_CPU_BATCH_NORM_CASE(CC_FLOAT32, float32)
		CC_CPU_BATCH_NORM_CASE(CC_FLOAT64, float64)
	default:
		UNSUPPORTED_DTYPE_LOG(dt);
		break;
	}
}

#define ARRAY_SC_OPS(op, oup, arr, elem, arrlen, dtype) \
	for (i = 0; i < arrlen; ++i) {                                    \
		*((dtype*)oup + i) = *((dtype*)arr + i) op *(dtype*)elem; \
	}

#define ARRAY_ELEM_SET(arr, elem, arrlen, dtype) \
	for (i = 0; i < arrlen; ++i) {              \
		*((dtype*)arr + i) = *(dtype*)elem; \
	}

#define ARRAY_ELEM_CLIP(arr, min, max, arrlen, dtype) \
	for (i = 0; i < arrlen; ++i) {                              \
		if (min) {                                          \
			*((dtype*)arr + i) =                        \
				*((dtype*)arr + i) < *(dtype*)min ? \
			*(dtype*)min : *((dtype*)arr + i);          \
		}                                                   \
		if (max) {                                          \
			*((dtype*)arr + i) =                        \
				*((dtype*)arr + i) > *(dtype*)max ? \
			*(dtype*)max : *((dtype*)arr + i);          \
		}                                                   \
	}

#define ARRAY_EW_OPS(op, oup, a, b, arrlen, dtype) \
	for (i = 0; i < arrlen; ++i) {                   \
		*((dtype*)oup + i) = *((dtype*)a + i) op \
		*((dtype*)b + i);                        \
	}

#define ARRAY_SUM(arr, arrlen, dtype, sum) \
	*(dtype*)sum = 0;                           \
	for (i = 0; i < arrlen; ++i) {              \
		*(dtype*)sum += *((dtype*)arr + i); \
	}

#define ARRAY_CAST_CASE(_DT, _srcdt, _dstdt) \
case _DT:                                                          \
	for (i = 0; i < arrlen; ++i)                               \
		*((_dstdt*)dst + i) = (_dstdt)*((_srcdt*)src + i); \
	break;

#define CC_CPU_ARRAY_CAST_IMPLEMENTATION(dtype) \
void cc_cpu_array_cast_ ## dtype(                                      \
	void *dst, const void *src, int arrlen, int dt)                \
{                                                                      \
	cc_int32 i;                                                    \
	switch (dt) {                                                  \
		ARRAY_CAST_CASE(CC_UINT8, cc_uint8, cc_ ## dtype);     \
		ARRAY_CAST_CASE(CC_UINT16, cc_uint16, cc_ ## dtype);   \
		ARRAY_CAST_CASE(CC_UINT32, cc_uint32, cc_ ## dtype);   \
		ARRAY_CAST_CASE(CC_UINT64, cc_uint64, cc_ ## dtype);   \
		ARRAY_CAST_CASE(CC_INT8, cc_int8, cc_ ## dtype);       \
		ARRAY_CAST_CASE(CC_INT16, cc_int16, cc_ ## dtype);     \
		ARRAY_CAST_CASE(CC_INT32, cc_int32, cc_ ## dtype);     \
		ARRAY_CAST_CASE(CC_INT64, cc_int64, cc_ ## dtype);     \
		ARRAY_CAST_CASE(CC_FLOAT32, cc_float32, cc_ ## dtype); \
		ARRAY_CAST_CASE(CC_FLOAT64, cc_float64, cc_ ## dtype); \
	default:                                                       \
		UNSUPPORTED_DTYPE_LOG(dt);                             \
		break;                                                 \
	}                                                              \
}

#define ARRAY_SET_CASE(_DT, _dt) \
case _DT:                                   \
	ARRAY_ELEM_SET(arr, x, arrlen, _dt) \
	break;
void cc_cpu_array_set(void *arr, int arrlen, const void *x, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_SET_CASE(CC_UINT8, cc_uint8);
		ARRAY_SET_CASE(CC_UINT16, cc_uint16);
		ARRAY_SET_CASE(CC_UINT32, cc_uint32);
		ARRAY_SET_CASE(CC_UINT64, cc_uint64);
		ARRAY_SET_CASE(CC_INT8, cc_int8);
		ARRAY_SET_CASE(CC_INT16, cc_int16);
		ARRAY_SET_CASE(CC_INT32, cc_int32);
		ARRAY_SET_CASE(CC_INT64, cc_int64);
		ARRAY_SET_CASE(CC_FLOAT32, cc_float32);
		ARRAY_SET_CASE(CC_FLOAT64, cc_float64);
	default:
		UNSUPPORTED_DTYPE_LOG(dt);
		break;
	}
}

#define ARRAY_CLIP_CASE(_DT, _dt) \
case _DT:                                     \
	ARRAY_ELEM_CLIP(arr, min, max, arrlen, _dt); \
	break;
void cc_cpu_array_clip_by_value(
	void *arr, int arrlen, const void *min, const void *max, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_CLIP_CASE(CC_UINT8, cc_uint8);
		ARRAY_CLIP_CASE(CC_UINT16, cc_uint16);
		ARRAY_CLIP_CASE(CC_UINT32, cc_uint32);
		ARRAY_CLIP_CASE(CC_UINT64, cc_uint64);
		ARRAY_CLIP_CASE(CC_INT8, cc_int8);
		ARRAY_CLIP_CASE(CC_INT16, cc_int16);
		ARRAY_CLIP_CASE(CC_INT32, cc_int32);
		ARRAY_CLIP_CASE(CC_INT64, cc_int64);
		ARRAY_CLIP_CASE(CC_FLOAT32, cc_float32);
		ARRAY_CLIP_CASE(CC_FLOAT64, cc_float64);
	default:
		UNSUPPORTED_DTYPE_LOG(dt);
		break;
	}
}

CC_CPU_ARRAY_CAST_IMPLEMENTATION  (uint8)
CC_CPU_ARRAY_CAST_IMPLEMENTATION  (uint16)
CC_CPU_ARRAY_CAST_IMPLEMENTATION  (uint32)
CC_CPU_ARRAY_CAST_IMPLEMENTATION  (uint64)
CC_CPU_ARRAY_CAST_IMPLEMENTATION  (int8)
CC_CPU_ARRAY_CAST_IMPLEMENTATION  (int16)
CC_CPU_ARRAY_CAST_IMPLEMENTATION  (int32)
CC_CPU_ARRAY_CAST_IMPLEMENTATION  (int64)
CC_CPU_ARRAY_CAST_IMPLEMENTATION  (float32)
CC_CPU_ARRAY_CAST_IMPLEMENTATION  (float64)

#define ARRAY_ADD_BY_CASE(_DT, _dt) \
case _DT:                                        \
	ARRAY_SC_OPS(+, oup, a, x, arrlen, _dt); \
	break;
void cc_cpu_array_add_by(void *oup,
	int arrlen, const void *a, const void *x, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_ADD_BY_CASE(CC_UINT8, cc_uint8);
		ARRAY_ADD_BY_CASE(CC_UINT16, cc_uint16);
		ARRAY_ADD_BY_CASE(CC_UINT32, cc_uint32);
		ARRAY_ADD_BY_CASE(CC_UINT64, cc_uint64);
		ARRAY_ADD_BY_CASE(CC_INT8, cc_int8);
		ARRAY_ADD_BY_CASE(CC_INT16, cc_int16);
		ARRAY_ADD_BY_CASE(CC_INT32, cc_int32);
		ARRAY_ADD_BY_CASE(CC_INT64, cc_int64);
		ARRAY_ADD_BY_CASE(CC_FLOAT32, cc_float32);
		ARRAY_ADD_BY_CASE(CC_FLOAT64, cc_float64);
	default:
		UNSUPPORTED_DTYPE_LOG(dt);
		break;
	}
}

#define ARRAY_SUB_BY_CASE(_DT, _dt) \
case _DT:                                        \
	ARRAY_SC_OPS(-, oup, a, x, arrlen, _dt); \
	break;
void cc_cpu_array_sub_by(void *oup,
	int arrlen, const void *a, const void *x, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_SUB_BY_CASE(CC_UINT8, cc_uint8);
		ARRAY_SUB_BY_CASE(CC_UINT16, cc_uint16);
		ARRAY_SUB_BY_CASE(CC_UINT32, cc_uint32);
		ARRAY_SUB_BY_CASE(CC_UINT64, cc_uint64);
		ARRAY_SUB_BY_CASE(CC_INT8, cc_int8);
		ARRAY_SUB_BY_CASE(CC_INT16, cc_int16);
		ARRAY_SUB_BY_CASE(CC_INT32, cc_int32);
		ARRAY_SUB_BY_CASE(CC_INT64, cc_int64);
		ARRAY_SUB_BY_CASE(CC_FLOAT32, cc_float32);
		ARRAY_SUB_BY_CASE(CC_FLOAT64, cc_float64);
	default:
		UNSUPPORTED_DTYPE_LOG(dt);
		break;
	}
}

#define ARRAY_MUL_BY_CASE(_DT, _dt) \
case _DT:                                        \
	ARRAY_SC_OPS(*, oup, a, x, arrlen, _dt); \
	break;
void cc_cpu_array_mul_by(void *oup,
	int arrlen, const void *a, const void *x, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_MUL_BY_CASE(CC_UINT8, cc_uint8);
		ARRAY_MUL_BY_CASE(CC_UINT16, cc_uint16);
		ARRAY_MUL_BY_CASE(CC_UINT32, cc_uint32);
		ARRAY_MUL_BY_CASE(CC_UINT64, cc_uint64);
		ARRAY_MUL_BY_CASE(CC_INT8, cc_int8);
		ARRAY_MUL_BY_CASE(CC_INT16, cc_int16);
		ARRAY_MUL_BY_CASE(CC_INT32, cc_int32);
		ARRAY_MUL_BY_CASE(CC_INT64, cc_int64);
		ARRAY_MUL_BY_CASE(CC_FLOAT32, cc_float32);
		ARRAY_MUL_BY_CASE(CC_FLOAT64, cc_float64);
	default:
		UNSUPPORTED_DTYPE_LOG(dt);
		break;
	}
}

#define ARRAY_DIV_BY_CASE(_DT, _dt) \
case _DT:                                        \
	ARRAY_SC_OPS(/, oup, a, x, arrlen, _dt); \
	break;
void cc_cpu_array_div_by(void *oup,
	int arrlen, const void *a, const void *x, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_DIV_BY_CASE(CC_UINT8, cc_uint8);
		ARRAY_DIV_BY_CASE(CC_UINT16, cc_uint16);
		ARRAY_DIV_BY_CASE(CC_UINT32, cc_uint32);
		ARRAY_DIV_BY_CASE(CC_UINT64, cc_uint64);
		ARRAY_DIV_BY_CASE(CC_INT8, cc_int8);
		ARRAY_DIV_BY_CASE(CC_INT16, cc_int16);
		ARRAY_DIV_BY_CASE(CC_INT32, cc_int32);
		ARRAY_DIV_BY_CASE(CC_INT64, cc_int64);
		ARRAY_DIV_BY_CASE(CC_FLOAT32, cc_float32);
		ARRAY_DIV_BY_CASE(CC_FLOAT64, cc_float64);
	default:
		UNSUPPORTED_DTYPE_LOG(dt);
		break;
	}
}

#define ARRAY_ADD_EW_CASE(_DT, _dt) \
case _DT:                                        \
	ARRAY_EW_OPS(+, oup, a, b, arrlen, _dt); \
	break;
void cc_cpu_array_add_ew(void *oup,
	int arrlen, const void *a, const void *b, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_ADD_EW_CASE(CC_UINT8, cc_uint8);
		ARRAY_ADD_EW_CASE(CC_UINT16, cc_uint16);
		ARRAY_ADD_EW_CASE(CC_UINT32, cc_uint32);
		ARRAY_ADD_EW_CASE(CC_UINT64, cc_uint64);
		ARRAY_ADD_EW_CASE(CC_INT8, cc_int8);
		ARRAY_ADD_EW_CASE(CC_INT16, cc_int16);
		ARRAY_ADD_EW_CASE(CC_INT32, cc_int32);
		ARRAY_ADD_EW_CASE(CC_INT64, cc_int64);
		ARRAY_ADD_EW_CASE(CC_FLOAT32, cc_float32);
		ARRAY_ADD_EW_CASE(CC_FLOAT64, cc_float64);
	default:
		UNSUPPORTED_DTYPE_LOG(dt);
		break;
	}
}

#define ARRAY_SUB_EW_CASE(_DT, _dt) \
case _DT:                                        \
	ARRAY_EW_OPS(-, oup, a, b, arrlen, _dt); \
	break;
void cc_cpu_array_sub_ew(void *oup,
	int arrlen, const void *a, const void *b, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_SUB_EW_CASE(CC_UINT8, cc_uint8);
		ARRAY_SUB_EW_CASE(CC_UINT16, cc_uint16);
		ARRAY_SUB_EW_CASE(CC_UINT32, cc_uint32);
		ARRAY_SUB_EW_CASE(CC_UINT64, cc_uint64);
		ARRAY_SUB_EW_CASE(CC_INT8, cc_int8);
		ARRAY_SUB_EW_CASE(CC_INT16, cc_int16);
		ARRAY_SUB_EW_CASE(CC_INT32, cc_int32);
		ARRAY_SUB_EW_CASE(CC_INT64, cc_int64);
		ARRAY_SUB_EW_CASE(CC_FLOAT32, cc_float32);
		ARRAY_SUB_EW_CASE(CC_FLOAT64, cc_float64);
	default:
		UNSUPPORTED_DTYPE_LOG(dt);
		break;
	}
}

#define ARRAY_MUL_EW_CASE(_DT, _dt) \
case _DT:                                        \
	ARRAY_EW_OPS(*, oup, a, b, arrlen, _dt); \
	break;
void cc_cpu_array_mul_ew(void *oup,
	int arrlen, const void *a, const void *b, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_MUL_EW_CASE(CC_UINT8, cc_uint8);
		ARRAY_MUL_EW_CASE(CC_UINT16, cc_uint16);
		ARRAY_MUL_EW_CASE(CC_UINT32, cc_uint32);
		ARRAY_MUL_EW_CASE(CC_UINT64, cc_uint64);
		ARRAY_MUL_EW_CASE(CC_INT8, cc_int8);
		ARRAY_MUL_EW_CASE(CC_INT16, cc_int16);
		ARRAY_MUL_EW_CASE(CC_INT32, cc_int32);
		ARRAY_MUL_EW_CASE(CC_INT64, cc_int64);
		ARRAY_MUL_EW_CASE(CC_FLOAT32, cc_float32);
		ARRAY_MUL_EW_CASE(CC_FLOAT64, cc_float64);
	default:
		UNSUPPORTED_DTYPE_LOG(dt);
		break;
	}
}

#define ARRAY_DIV_EW_CASE(_DT, _dt) \
case _DT:                                        \
	ARRAY_EW_OPS(/, oup, a, b, arrlen, _dt); \
	break;
void cc_cpu_array_div_ew(void *oup,
	int arrlen, const void *a, const void *b, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_DIV_EW_CASE(CC_UINT8, cc_uint8);
		ARRAY_DIV_EW_CASE(CC_UINT16, cc_uint16);
		ARRAY_DIV_EW_CASE(CC_UINT32, cc_uint32);
		ARRAY_DIV_EW_CASE(CC_UINT64, cc_uint64);
		ARRAY_DIV_EW_CASE(CC_INT8, cc_int8);
		ARRAY_DIV_EW_CASE(CC_INT16, cc_int16);
		ARRAY_DIV_EW_CASE(CC_INT32, cc_int32);
		ARRAY_DIV_EW_CASE(CC_INT64, cc_int64);
		ARRAY_DIV_EW_CASE(CC_FLOAT32, cc_float32);
		ARRAY_DIV_EW_CASE(CC_FLOAT64, cc_float64);
	default:
		UNSUPPORTED_DTYPE_LOG(dt);
		break;
	}
}

#define ARRAY_DOTPROD_CASE(_DT, _dt) \
case _DT:                                                  \
*((_dt*)x) = 0;                                            \
for (i = 0; i < arrlen; ++i)                               \
	*((_dt*)x) += *(((_dt*)a) + i) * *(((_dt*)b) + i); \
break;
void cc_cpu_array_dot_prod(
	const void *a, const void *b, int arrlen, void *x, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_DOTPROD_CASE(CC_UINT8, cc_uint8);
		ARRAY_DOTPROD_CASE(CC_UINT16, cc_uint16);
		ARRAY_DOTPROD_CASE(CC_UINT32, cc_uint32);
		ARRAY_DOTPROD_CASE(CC_UINT64, cc_uint64);
		ARRAY_DOTPROD_CASE(CC_INT8, cc_int8);
		ARRAY_DOTPROD_CASE(CC_INT16, cc_int16);
		ARRAY_DOTPROD_CASE(CC_INT32, cc_int32);
		ARRAY_DOTPROD_CASE(CC_INT64, cc_int64);
		ARRAY_DOTPROD_CASE(CC_FLOAT32, cc_float32);
		ARRAY_DOTPROD_CASE(CC_FLOAT64, cc_float64);
	default:
		UNSUPPORTED_DTYPE_LOG(dt);
		break;
	}
}

#define ARRAY_SUM_CASE(_DT, _dt) \
case _DT:                               \
	ARRAY_SUM(arr, arrlen, _dt, x); \
	break;
void cc_cpu_array_sum(const void *arr, int arrlen, void *x, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_SUM_CASE(CC_UINT8, cc_uint8);
		ARRAY_SUM_CASE(CC_UINT16, cc_uint16);
		ARRAY_SUM_CASE(CC_UINT32, cc_uint32);
		ARRAY_SUM_CASE(CC_UINT64, cc_uint64);
		ARRAY_SUM_CASE(CC_INT8, cc_int8);
		ARRAY_SUM_CASE(CC_INT16, cc_int16);
		ARRAY_SUM_CASE(CC_INT32, cc_int32);
		ARRAY_SUM_CASE(CC_INT64, cc_int64);
		ARRAY_SUM_CASE(CC_FLOAT32, cc_float32);
		ARRAY_SUM_CASE(CC_FLOAT64, cc_float64);
	default:
		UNSUPPORTED_DTYPE_LOG(dt);
		break;
	}
}

#define ARRAY_MEAN_CASE(_DT, _dt) \
case _DT:                               \
	ARRAY_SUM(arr, arrlen, _dt, x); \
	*(_dt*)x /= arrlen;             \
	break;
void cc_cpu_array_mean(const void *arr, int arrlen, void *x, int dt)
{
	cc_int32 i;
	switch (dt) {
		ARRAY_MEAN_CASE(CC_UINT8, cc_uint8);
		ARRAY_MEAN_CASE(CC_UINT16, cc_uint16);
		ARRAY_MEAN_CASE(CC_UINT32, cc_uint32);
		ARRAY_MEAN_CASE(CC_UINT64, cc_uint64);
		ARRAY_MEAN_CASE(CC_INT8, cc_int8);
		ARRAY_MEAN_CASE(CC_INT16, cc_int16);
		ARRAY_MEAN_CASE(CC_INT32, cc_int32);
		ARRAY_MEAN_CASE(CC_INT64, cc_int64);
		ARRAY_MEAN_CASE(CC_FLOAT32, cc_float32);
		ARRAY_MEAN_CASE(CC_FLOAT64, cc_float64);
	default:
		UNSUPPORTED_DTYPE_LOG(dt);
		break;
	}
}

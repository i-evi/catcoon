#include <stdlib.h>
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
		utlog_format(UTLOG_ERR,
			"cc_cpufn: unsupported dtype %x\n", dt);
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
		utlog_format(UTLOG_ERR,
			"cc_cpufn: unsupported dtype %x\n", dt);
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
		utlog_format(UTLOG_ERR,
			"cc_cpufn: unsupported dtype %x\n", dt);
	}
}

#define CC_CPU_MAX_POOL2D_CASE(DT, dtype) \
case DT:                                                             \
for (i = 0; i < o_y; ++i) {                                          \
	for (j = 0; j < o_x; ++j)                                    \
	{                                                            \
		*(dtype*)v_max = *((dtype*)inp + s * i * x + s * j); \
		for (k = 0; k < s; ++k)                              \
		{                                                    \
			for (l = 0; l < s; ++l)                      \
			{                                            \
				curr = (dtype*)inp +                 \
					(s * i + k) * x + j * s + l; \
				*(dtype*)v_max = *((dtype*)curr) >   \
					*(dtype*)v_max ?             \
				*((dtype*)curr) : *(dtype*)v_max;    \
			}                                            \
		}                                                    \
		*((dtype*)oup + i * o_x + j) = *(dtype*)v_max;       \
	}                                                            \
}                                                                    \
break;

void cc_cpu_max_pool2d(const void *inp, void *oup,
	cc_int32 x, cc_int32 y, cc_int32 s, cc_dtype dt)
{
	cc_int32 o_x = x / s;
	cc_int32 o_y = y / s;
	cc_int32 i, j, k, l;
	void *curr, *v_max;
	cc_assert_alloc(v_max = malloc(cc_dtype_size(dt)));
	switch (dt) {
	CC_CPU_MAX_POOL2D_CASE(CC_UINT8,   cc_uint8);
	CC_CPU_MAX_POOL2D_CASE(CC_UINT16,  cc_uint16);
	CC_CPU_MAX_POOL2D_CASE(CC_UINT32,  cc_uint32);
	CC_CPU_MAX_POOL2D_CASE(CC_UINT64,  cc_uint64);
	CC_CPU_MAX_POOL2D_CASE(CC_INT8,    cc_int8);
	CC_CPU_MAX_POOL2D_CASE(CC_INT16,   cc_int16);
	CC_CPU_MAX_POOL2D_CASE(CC_INT32,   cc_int32);
	CC_CPU_MAX_POOL2D_CASE(CC_INT64,   cc_int64);
	CC_CPU_MAX_POOL2D_CASE(CC_FLOAT32, cc_float32);
	CC_CPU_MAX_POOL2D_CASE(CC_FLOAT64, cc_float64);
	default:
		utlog_format(UTLOG_ERR,
			"cc_cpufn: unsupported dtype %x\n", dt);
	}
	free(v_max);
}

#define CC_CPU_CONV2D_IMPLEMENTATION(dt) \
static void cc_cpu_conv2d_ ## dt (cc_ ## dt *inp, cc_ ## dt *oup, \
	cc_int32 x, cc_int32 y, cc_int32 oup_x, cc_int32 oup_y,   \
	cc_int32 sx, cc_int32 sy, cc_ ## dt *filter, cc_int32 fw) \
{                                                                 \
  cc_int32 i, j, k, l, oup_i, oup_j;                              \
  cc_int32 half_fl = fw >> 1;                                     \
  cc_ ## dt sum;                                                  \
    for (i = half_fl; i < y - half_fl; i += sy) {                 \
      for (j = half_fl; j < x - half_fl; j += sx) {               \
        sum = 0;                                                  \
        for (k = -half_fl; k <= half_fl; ++k) {                   \
          for (l = -half_fl; l <= half_fl; ++l) {                 \
            sum += *(inp + (i + k) * x + (j + l)) *               \
                *(filter + (k + half_fl) * fw + (l + half_fl));   \
        }                                                         \
      }                                                           \
      oup_i = ((i - half_fl) / sy);                               \
      oup_j = ((j - half_fl) / sx);                               \
      *(oup + oup_i * oup_x + oup_j) = sum;                       \
    }                                                             \
  }                                                               \
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

void cc_cpu_conv2d(const void *inp, void *oup, cc_int32 x, cc_int32 y,
		cc_int32 oup_x, cc_int32 oup_y, cc_int32 sx, cc_int32 sy,
	const void *filter, cc_int32 fw, cc_dtype dt)
{
	switch (dt) {
	case CC_UINT8:
		cc_cpu_conv2d_uint8((cc_uint8*)inp,
			(cc_uint8*)oup, x, y, oup_x, oup_y,
			sx, sy, (cc_uint8*)filter, fw);
		break;
	case CC_UINT16:
		cc_cpu_conv2d_uint16((cc_uint16*)inp,
			(cc_uint16*)oup, x, y, oup_x, oup_y,
			sx, sy, (cc_uint16*)filter, fw);
		break;
	case CC_UINT32:
		cc_cpu_conv2d_uint32((cc_uint32*)inp,
			(cc_uint32*)oup, x, y, oup_x, oup_y,
			sx, sy, (cc_uint32*)filter, fw);
		break;
	case CC_UINT64:
		cc_cpu_conv2d_uint64((cc_uint64*)inp,
			(cc_uint64*)oup, x, y, oup_x, oup_y,
			sx, sy, (cc_uint64*)filter, fw);
		break;
	case CC_INT8:
		cc_cpu_conv2d_int8((cc_int8*)inp,
			(cc_int8*)oup, x, y, oup_x, oup_y,
			sx, sy, (cc_int8*)filter, fw);
		break;
	case CC_INT16:
		cc_cpu_conv2d_int16((cc_int16*)inp,
			(cc_int16*)oup, x, y, oup_x, oup_y,
			sx, sy, (cc_int16*)filter, fw);
		break;
	case CC_INT32:
		cc_cpu_conv2d_int32((cc_int32*)inp,
			(cc_int32*)oup, x, y, oup_x, oup_y,
			sx, sy, (cc_int32*)filter, fw);
		break;
	case CC_INT64:
		cc_cpu_conv2d_int64((cc_int64*)inp,
			(cc_int64*)oup, x, y, oup_x, oup_y,
			sx, sy, (cc_int64*)filter, fw);
		break;
	case CC_FLOAT32:
		cc_cpu_conv2d_float32((cc_float32*)inp,
			(cc_float32*)oup, x, y, oup_x, oup_y,
			sx, sy, (cc_float32*)filter, fw);
		break;
	case CC_FLOAT64:
		cc_cpu_conv2d_float64((cc_float64*)inp,
			(cc_float64*)oup, x, y, oup_x, oup_y,
			sx, sy, (cc_float64*)filter, fw);
		break;
	default:
		utlog_format(UTLOG_ERR,
			"cc_cpufn: unsupported dtype %x\n", dt);
	}
}

#define CC_CPU_FULLY_CONNECTED_PROCESS \
for (i = 0; i < ow; ++i) {                                    \
	oup[i] = 0;                                           \
	for (j = 0; j < iw; ++j) {                            \
		oup[i] += (*(inp + j)) * (*(w + iw * i + j)); \
	}                                                     \
	if (b)                                                \
		oup[i] += b[i];                               \
}

static void cc_cpu_fully_connected_float32(
	cc_float32 *inp, cc_float32 *oup, cc_float32 *w,
	cc_float32 *b, cc_int32 iw, cc_int32 ow)
{
	cc_int32 i, j;
#ifdef ENABLE_OPENMP
#pragma omp parallel for private(i, j)
#endif
	CC_CPU_FULLY_CONNECTED_PROCESS;
}

static void cc_cpu_fully_connected_float64(
	cc_float64 *inp, cc_float64 *oup, cc_float64 *w,
	cc_float64 *b, cc_int32 iw, cc_int32 ow)
{
	cc_int32 i, j;
#ifdef ENABLE_OPENMP
#pragma omp parallel for private(i, j)
#endif
	CC_CPU_FULLY_CONNECTED_PROCESS;
}

void cc_cpu_fully_connected(const void *inp,
		void *oup, const void *w, const void *b,
	cc_int32 iw, cc_int32 ow, cc_dtype dt)
{
	switch (dt) {
	case CC_FLOAT32:
		cc_cpu_fully_connected_float32(
			(cc_float32*)inp, (cc_float32*)oup,
			(cc_float32*)w, (cc_float32*)b, iw, ow);
		break;
	case CC_FLOAT64:
		cc_cpu_fully_connected_float64(
			(cc_float64*)inp, (cc_float64*)oup,
			(cc_float64*)w, (cc_float64*)b, iw, ow);
		break;
	default:
		UNSUPPORTED_DTYPE_LOG(dt);
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

void cc_cpu_batch_norm(void *inp,
	cc_int32 len, const void *bnpara, cc_dtype dt)
{
	switch (dt) {
	case CC_UINT8:
		cc_cpu_batch_norm_uint8(
			(cc_uint8*)inp, len, (cc_uint8*)bnpara);
		break;
	case CC_UINT16:
		cc_cpu_batch_norm_uint16(
			(cc_uint16*)inp, len, (cc_uint16*)bnpara);
		break;
	case CC_UINT32:
		cc_cpu_batch_norm_uint32(
			(cc_uint32*)inp, len, (cc_uint32*)bnpara);
		break;
	case CC_UINT64:
		cc_cpu_batch_norm_uint64(
			(cc_uint64*)inp, len, (cc_uint64*)bnpara);
		break;
	case CC_INT8:
		cc_cpu_batch_norm_int8(
			(cc_int8*)inp, len, (cc_int8*)bnpara);
		break;
	case CC_INT16:
		cc_cpu_batch_norm_int16(
			(cc_int16*)inp, len, (cc_int16*)bnpara);
		break;
	case CC_INT32:
		cc_cpu_batch_norm_int32(
			(cc_int32*)inp, len, (cc_int32*)bnpara);
		break;
	case CC_INT64:
		cc_cpu_batch_norm_int64(
			(cc_int64*)inp, len, (cc_int64*)bnpara);
		break;
	case CC_FLOAT32:
		cc_cpu_batch_norm_float32(
			(cc_float32*)inp, len, (cc_float32*)bnpara);
		break;
	case CC_FLOAT64:
		cc_cpu_batch_norm_float64(
			(cc_float64*)inp, len, (cc_float64*)bnpara);
		break;
	default:
		utlog_format(UTLOG_ERR,
			"cc_cpufn: unsupported dtype %x\n", dt);
	}
}
